import click
from peewee import fn
from playhouse.postgres_ext import *
from loguru import logger
from models import ImageHash, PipelineBatch, PipelineBatchItem, Detection
from datetime import datetime
import pickle
import os
from const import CPUS_LIMIT
import time
from collections import defaultdict


class DedupedHash(Model):
    id = AutoField()
    hash_id = IntegerField(index=True)  # original ImageHash pk
    group_id = IntegerField(index=True)  # dedupe group
    pipeline_batch_item = ForeignKeyField(
        PipelineBatchItem, field="id_pipeline_batch_item", index=True
    )
    detection = ForeignKeyField(Detection, field="id_detection", index=True)
    scan_filename = CharField()
    image_hash = CharField(index=True)
    created = DateTimeField(constraints=[SQL("DEFAULT CURRENT_TIMESTAMP")])

    class Meta:
        table_name = "deduped_hash"
        database = ImageHash._meta.database


@click.command("step06-dedupe-hashes")
@click.option("--id-pipeline-run", type=int, required=True, help="Pipeline run to deduplicate")
@click.option(
    "--hamming-threshold",
    type=int,
    default=8,
    show_default=True,
    help="Hamming distance threshold for fuzzy matching (0 = exact match only)",
)
@click.option(
    "--shard-id",
    type=int,
    default=None,
    help="Process only this shard (0-based index). Use with --total-shards for parallel processing",
)
@click.option(
    "--total-shards",
    type=int,
    default=1,
    show_default=True,
    help="Total number of shards to split dataset into",
)
@click.option(
    "--max-hashes-per-shard",
    type=int,
    default=10_000_000,
    show_default=True,
    help="Maximum hashes per shard (auto-shard if exceeded)",
)
@click.option(
    "--checkpoint-file",
    type=str,
    default="dedupe_hash_checkpoint.pkl",
    help="File to save progress for resuming",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from checkpoint file",
)
def step06_dedupe_hash(
    id_pipeline_run,
    hamming_threshold,
    shard_id,
    total_shards,
    max_hashes_per_shard,
    checkpoint_file,
    resume,
):
    """
    Deduplicate image hashes using exact or fuzzy matching.

    For large datasets (>10M hashes), use sharding:

    \b
    # Auto-shard (sequential):
    python pipeline.py step06-dedupe-hashes --id-pipeline-run=1

    \b
    # Manual parallel sharding (run these in parallel):
    python pipeline.py step06-dedupe-hashes --id-pipeline-run=1 --shard-id=0 --total-shards=10 &
    python pipeline.py step06-dedupe-hashes --id-pipeline-run=1 --shard-id=1 --total-shards=10 &
    ...
    """

    logger.info("Starting hash-based deduplication...")

    # Ensure DedupedHash table exists
    db = ImageHash._meta.database
    db.create_tables([DedupedHash], safe=True)

    # Count total hashes
    pipeline_batches = list(
        PipelineBatch.select().where(PipelineBatch.pipeline_run == id_pipeline_run)
    )
    pb_item_ids = [item.id_pipeline_batch_item for pb in pipeline_batches for item in pb.items]

    total_hashes = ImageHash.select().where(ImageHash.pipeline_batch_item.in_(pb_item_ids)).count()

    logger.info(f"Total hashes in pipeline run {id_pipeline_run}: {total_hashes:,}")

    # Auto-shard if needed
    if shard_id is None and total_hashes > max_hashes_per_shard:
        required_shards = (total_hashes // max_hashes_per_shard) + 1
        logger.info(f"Dataset exceeds {max_hashes_per_shard:,} hashes per shard")
        logger.info(f"Auto-sharding into {required_shards} shards")
        logger.info("=" * 70)
        logger.info("For faster processing, run shards in parallel:")
        logger.info("")
        for i in range(required_shards):
            logger.info(
                f"  python pipeline.py step06-dedupe-hashes --id-pipeline-run={id_pipeline_run} "
                f"--shard-id={i} --total-shards={required_shards} &"
            )
        logger.info("")
        logger.info("=" * 70)
        logger.info("Processing all shards sequentially...")

        for i in range(required_shards):
            logger.info(f"\n{'='*70}")
            logger.info(f"SHARD {i+1}/{required_shards}")
            logger.info(f"{'='*70}\n")

            shard_checkpoint = f"dedupe_hash_checkpoint_shard{i}_of_{required_shards}.pkl"

            _process_single_shard(
                id_pipeline_run=id_pipeline_run,
                pb_item_ids=pb_item_ids,
                shard_id=i,
                total_shards=required_shards,
                hamming_threshold=hamming_threshold,
                checkpoint_file=shard_checkpoint,
                resume=resume,
            )

        logger.info(f"\n{'='*70}")
        logger.info(f"✓ ALL {required_shards} SHARDS COMPLETED")
        logger.info(f"{'='*70}\n")
        return

    # Process single shard (or entire dataset if small)
    if shard_id is not None:
        logger.info(f"Processing shard {shard_id + 1}/{total_shards}")
        checkpoint_file = f"dedupe_hash_checkpoint_shard{shard_id}_of_{total_shards}.pkl"

    _process_single_shard(
        id_pipeline_run=id_pipeline_run,
        pb_item_ids=pb_item_ids,
        shard_id=shard_id,
        total_shards=total_shards,
        hamming_threshold=hamming_threshold,
        checkpoint_file=checkpoint_file,
        resume=resume,
    )


def _process_single_shard(
    id_pipeline_run,
    pb_item_ids,
    shard_id,
    total_shards,
    hamming_threshold,
    checkpoint_file,
    resume,
):
    """Process a single shard of hashes"""

    logger.info(
        f"Starting hash deduplication for pipeline run {id_pipeline_run} "
        f"(Hamming threshold {hamming_threshold})"
    )

    if shard_id is not None:
        logger.info(f"Shard: {shard_id + 1}/{total_shards}")

    # Get hashes for this shard
    t0 = time.time()
    logger.info("Loading hash metadata...")

    hashes_meta = {}
    query = ImageHash.select(
        ImageHash.id_imagehash,
        ImageHash.pipeline_batch_item,
        ImageHash.detection,
        ImageHash.scan_filename,
        ImageHash.image_hash,
    ).where(ImageHash.pipeline_batch_item.in_(pb_item_ids))

    # Add shard filter if sharding
    if shard_id is not None and total_shards > 1:
        query = query.where((ImageHash.id_imagehash % total_shards) == shard_id)

    for h in query.dicts():
        hashes_meta[h["id_imagehash"]] = {
            "pipeline_batch_item": h["pipeline_batch_item"],
            "detection": h["detection"],
            "scan_filename": h["scan_filename"],
            "image_hash": h["image_hash"],
        }

    logger.info(f"Found {len(hashes_meta):,} hashes to process in {time.time() - t0:.1f}s")

    if not hashes_meta:
        logger.warning("No hashes found to deduplicate.")
        return

    # Find similar pairs
    if hamming_threshold == 0:
        similar_pairs = _find_exact_matches(hashes_meta)
    else:
        similar_pairs = _find_fuzzy_matches(hashes_meta, hamming_threshold, checkpoint_file, resume)

    logger.info(f"Found {len(similar_pairs):,} similar pairs")

    # Cluster using Union-Find
    logger.info("Clustering hashes using Union-Find...")
    dedupe_assignments, num_groups = _cluster_hashes(hashes_meta.keys(), similar_pairs)
    logger.info(f"Assigned {len(dedupe_assignments):,} hashes to {num_groups:,} dedupe groups.")

    # Write assignments
    _write_hash_assignments_batched(
        hashes_meta, dedupe_assignments, pb_item_ids, shard_id, total_shards
    )

    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info(f"Removed checkpoint file: {checkpoint_file}")

    logger.info(
        f"✓ Hash deduplication complete: {len(dedupe_assignments):,} hashes in {num_groups:,} groups."
    )


def _find_exact_matches(hashes_meta):
    """Find exact hash matches - very fast"""
    logger.info("Finding exact hash matches...")

    # Group by hash value
    hash_groups = defaultdict(list)
    for hash_id, meta in hashes_meta.items():
        hash_groups[meta["image_hash"]].append(hash_id)

    # Create pairs from groups
    similar_pairs = []
    for hash_value, hash_ids in hash_groups.items():
        if len(hash_ids) > 1:
            # Create all pairs within this group
            for i in range(len(hash_ids)):
                for j in range(i + 1, len(hash_ids)):
                    pair = tuple(sorted([hash_ids[i], hash_ids[j]]))
                    similar_pairs.append(pair)

    logger.info(f"Found {len(hash_groups):,} unique hash values")
    logger.info(f"Found {len(similar_pairs):,} exact match pairs")

    return similar_pairs


def _find_fuzzy_matches(hashes_meta, hamming_threshold, checkpoint_file, resume):
    """Find fuzzy matches using Hamming distance"""
    logger.info(f"Finding fuzzy matches (Hamming distance <= {hamming_threshold})...")

    # Load checkpoint if resuming
    if resume and os.path.exists(checkpoint_file):
        logger.info(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
            similar_pairs = list(checkpoint.get("similar_pairs", []))
            processed_idx = checkpoint.get("processed_idx", 0)
        logger.info(f"Resumed: {len(similar_pairs)} pairs found, processed {processed_idx} hashes")
    else:
        similar_pairs = []
        processed_idx = 0

    hash_ids = list(hashes_meta.keys())
    total_hashes = len(hash_ids)

    # Convert hex strings to integers for faster comparison
    hash_values = {}
    for hash_id in hash_ids:
        hex_str = hashes_meta[hash_id]["image_hash"]
        try:
            hash_values[hash_id] = int(hex_str, 16)
        except ValueError:
            logger.warning(f"Invalid hash format for ID {hash_id}: {hex_str}")
            hash_values[hash_id] = 0

    logger.info(f"Comparing {total_hashes:,} hashes (this may take a while)...")

    checkpoint_interval = max(1000, total_hashes // 100)
    last_checkpoint_time = time.time()

    for i in range(processed_idx, total_hashes):
        hash_id_i = hash_ids[i]
        hash_val_i = hash_values[hash_id_i]

        # Compare with all subsequent hashes
        for j in range(i + 1, total_hashes):
            hash_id_j = hash_ids[j]
            hash_val_j = hash_values[hash_id_j]

            # Calculate Hamming distance using XOR and popcount
            distance = bin(hash_val_i ^ hash_val_j).count("1")

            if distance <= hamming_threshold:
                pair = tuple(sorted([hash_id_i, hash_id_j]))
                similar_pairs.append(pair)

        # Progress logging
        if (i + 1) % 1000 == 0:
            progress = 100 * (i + 1) / total_hashes
            logger.info(
                f"Progress: {i + 1:,}/{total_hashes:,} ({progress:.1f}%) - "
                f"Found {len(similar_pairs):,} pairs"
            )

        # Checkpoint periodically
        if (i + 1) % checkpoint_interval == 0 or time.time() - last_checkpoint_time > 300:
            _save_hash_checkpoint(checkpoint_file, similar_pairs, i + 1)
            last_checkpoint_time = time.time()

    # Final checkpoint
    _save_hash_checkpoint(checkpoint_file, similar_pairs, total_hashes)

    logger.info(f"Found {len(similar_pairs):,} fuzzy match pairs")
    return similar_pairs


def _save_hash_checkpoint(checkpoint_file, similar_pairs, processed_idx):
    """Save progress to checkpoint file"""
    with open(checkpoint_file, "wb") as f:
        pickle.dump(
            {
                "similar_pairs": similar_pairs,
                "processed_idx": processed_idx,
            },
            f,
        )
    logger.debug(f"Checkpoint saved: {len(similar_pairs)} pairs, processed {processed_idx} hashes")


def _cluster_hashes(hash_ids, similar_pairs):
    """Use Union-Find algorithm to cluster hashes into connected components"""
    # Initialize parent pointers
    parent = {hid: hid for hid in hash_ids}

    def find(x):
        """Find root with path compression"""
        path = []
        while parent[x] != x:
            path.append(x)
            x = parent[x]
        # Path compression
        for node in path:
            parent[node] = x
        return x

    def union(x, y):
        """Union two sets"""
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    # Build connected components
    logger.info("Computing duplicate groups...")
    for i, (id1, id2) in enumerate(similar_pairs):
        union(id1, id2)
        if (i + 1) % 100000 == 0:
            logger.info(f"Processed {i + 1:,}/{len(similar_pairs):,} pairs")

    # Gather groups
    logger.info("Gathering groups...")
    group_map = {}
    for hid in hash_ids:
        root = find(hid)
        if root not in group_map:
            group_map[root] = []
        group_map[root].append(hid)

    # Assign group IDs (using min of group as ID for consistency)
    logger.info("Assigning group IDs...")
    dedupe_assignments = {}
    for i, (root, group_members) in enumerate(group_map.items()):
        group_id = min(group_members)  # Use min ID as group identifier
        for hid in group_members:
            dedupe_assignments[hid] = group_id

        if (i + 1) % 10000 == 0:
            logger.info(f"Assigned {i + 1:,}/{len(group_map):,} groups")

    num_groups = len(group_map)
    logger.info(f"Found {num_groups:,} unique groups from {len(hash_ids):,} hashes.")

    return dedupe_assignments, num_groups


def _write_hash_assignments_batched(
    hashes_meta, dedupe_assignments, pb_item_ids, shard_id, total_shards
):
    """Write deduplication assignments to database in batches"""
    db = DedupedHash._meta.database
    db.create_tables([DedupedHash], safe=True)

    # When sharding, only delete assignments for this shard's hashes
    logger.info("Clearing old assignments for this shard...")
    hash_ids_to_delete = list(dedupe_assignments.keys())

    if hash_ids_to_delete:
        deleted = DedupedHash.delete().where(DedupedHash.hash_id.in_(hash_ids_to_delete)).execute()
        logger.info(f"Deleted {deleted} existing assignments")
    else:
        logger.info("No existing assignments to delete")

    logger.info("Writing new assignments...")
    now = datetime.utcnow()

    chunk_size = 5000
    hash_ids = list(dedupe_assignments.keys())
    total_written = 0

    with db.atomic():
        for i in range(0, len(hash_ids), chunk_size):
            chunk_ids = hash_ids[i : i + chunk_size]

            # Prepare rows
            output_rows = []
            for hid in chunk_ids:
                meta = hashes_meta[hid]

                output_rows.append(
                    {
                        "hash_id": hid,
                        "group_id": dedupe_assignments[hid],
                        "pipeline_batch_item": meta["pipeline_batch_item"],
                        "detection": meta["detection"],
                        "scan_filename": meta["scan_filename"],
                        "image_hash": meta["image_hash"],
                        "created": now,
                    }
                )

            # Batch insert
            if output_rows:
                DedupedHash.insert_many(output_rows).execute()
                total_written += len(output_rows)
                logger.info(
                    f"Wrote {total_written:,}/{len(hash_ids):,} assignments "
                    f"({100*total_written/len(hash_ids):.1f}%)"
                )

    logger.info(f"✓ Successfully wrote {total_written:,} deduplicated hashes")
