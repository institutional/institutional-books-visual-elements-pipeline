import click
from peewee import fn, SQL
from playhouse.postgres_ext import *
from loguru import logger
from models import ImageHash, PipelineBatch, PipelineBatchItem, Detection
from datetime import datetime
import pickle
import os
import h5py
import numpy as np
from const import (
    CPUS_LIMIT,
    HASH_DB_CHUNK_SIZE,
    HASH_DEDUPE_HAMMING_THRESHOLD,
    HASH_DEDUPE_TOTAL_SHARDS,
    HASH_DEDUPE_MAX_HASHES_PER_SHARD,
)
import time
from collections import defaultdict
import multiprocessing as mp
from functools import partial


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
    default=HASH_DEDUPE_HAMMING_THRESHOLD,
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
    default=HASH_DEDUPE_TOTAL_SHARDS,
    show_default=True,
    help="Total number of shards to split dataset into",
)
@click.option(
    "--max-hashes-per-shard",
    type=int,
    default=HASH_DEDUPE_MAX_HASHES_PER_SHARD,
    show_default=True,
    help="Maximum hashes per shard (auto-shard if exceeded)",
)
@click.option(
    "--workers",
    type=int,
    default=CPUS_LIMIT,
    show_default=True,
    help="Number of parallel workers for hash comparison",
)
@click.option(
    "--cache-dir",
    type=str,
    default="./hash_cache",
    help="Directory to cache hash data files",
)
@click.option(
    "--force-reload",
    is_flag=True,
    default=False,
    help="Force reload hashes from database (ignore cache)",
)
def step06_dedupe_hash(
    id_pipeline_run,
    hamming_threshold,
    shard_id,
    total_shards,
    max_hashes_per_shard,
    workers,
    cache_dir,
    force_reload,
):
    """
    Deduplicate image hashes using exact or fuzzy matching.

    For large datasets (>10M hashes), use sharding.
    """

    logger.info("Starting hash-based deduplication...")

    # Ensure DedupedHash table exists
    db = ImageHash._meta.database
    db.create_tables([DedupedHash], safe=True)

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Get pipeline batches for this run
    pipeline_batches = list(
        PipelineBatch.select().where(PipelineBatch.pipeline_run == id_pipeline_run)
    )

    if not pipeline_batches:
        logger.error(f"No pipeline batches found for run {id_pipeline_run}")
        return

    # Get pipeline batch IDs
    pb_ids = [pb.id_pipeline_batch for pb in pipeline_batches]
    logger.info(f"Found {len(pb_ids)} pipeline batches")

    # Query pipeline batch items
    pb_items = list(
        PipelineBatchItem.select(PipelineBatchItem.id_pipeline_batch_item).where(
            PipelineBatchItem.pipeline_batch.in_(pb_ids)
        )
    )

    if not pb_items:
        logger.error(f"No pipeline batch items found for run {id_pipeline_run}")
        return

    pb_item_ids = [item.id_pipeline_batch_item for item in pb_items]

    logger.info(f"Found {len(pb_item_ids)} pipeline batch items")

    # Count total hashes using subquery to avoid large IN clause
    total_hashes = (
        ImageHash.select()
        .join(
            PipelineBatchItem,
            on=(ImageHash.pipeline_batch_item == PipelineBatchItem.id_pipeline_batch_item),
        )
        .where(PipelineBatchItem.pipeline_batch.in_(pb_ids))
        .count()
    )

    logger.info(f"Total hashes in pipeline run {id_pipeline_run}: {total_hashes:,}")

    if total_hashes == 0:
        logger.warning("No hashes found for this pipeline run")
        return

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

            _process_single_shard(
                id_pipeline_run=id_pipeline_run,
                pb_ids=pb_ids,
                shard_id=i,
                total_shards=required_shards,
                hamming_threshold=hamming_threshold,
                workers=workers,
                cache_dir=cache_dir,
                force_reload=force_reload,
            )

        logger.info(f"\n{'='*70}")
        logger.info(f"✓ ALL {required_shards} SHARDS COMPLETED")
        logger.info(f"{'='*70}\n")
        return

    # Process single shard (or entire dataset if small)
    _process_single_shard(
        id_pipeline_run=id_pipeline_run,
        pb_ids=pb_ids,
        shard_id=shard_id,
        total_shards=total_shards,
        hamming_threshold=hamming_threshold,
        workers=workers,
        cache_dir=cache_dir,
        force_reload=force_reload,
    )


def _process_single_shard(
    id_pipeline_run,
    pb_ids,
    shard_id,
    total_shards,
    hamming_threshold,
    workers,
    cache_dir,
    force_reload,
):
    """Process a single shard of hashes"""

    logger.info(
        f"Starting hash deduplication for pipeline run {id_pipeline_run} "
        f"(Hamming threshold {hamming_threshold})"
    )

    if shard_id is not None:
        logger.info(f"Shard: {shard_id + 1}/{total_shards}")

    # Generate cache filename
    if shard_id is not None:
        cache_file = os.path.join(
            cache_dir, f"hashes_run{id_pipeline_run}_shard{shard_id}_of_{total_shards}.h5"
        )
    else:
        cache_file = os.path.join(cache_dir, f"hashes_run{id_pipeline_run}.h5")

    # Load or create hash data file
    if os.path.exists(cache_file) and not force_reload:
        logger.info(f"Loading hashes from cache: {cache_file}")
        hashes_meta, hash_ids, hash_values_array = _load_hashes_from_file(cache_file)
    else:
        logger.info(f"Loading hashes from database and saving to: {cache_file}")
        hashes_meta, hash_ids, hash_values_array = _load_and_save_hashes(
            pb_ids, shard_id, total_shards, cache_file
        )

    logger.info(f"Loaded {len(hash_ids):,} hashes")

    if not hash_ids:
        logger.warning("No hashes found to deduplicate.")
        return

    # Find similar pairs
    if hamming_threshold == 0:
        similar_pairs = _find_exact_matches(hashes_meta)
    else:
        similar_pairs = _find_fuzzy_matches_parallel(
            hash_ids, hash_values_array, hamming_threshold, workers
        )

    logger.info(f"Found {len(similar_pairs):,} similar pairs")

    # Cluster using Union-Find
    logger.info("Clustering hashes using Union-Find...")
    dedupe_assignments, num_groups = _cluster_hashes(hash_ids, similar_pairs)
    logger.info(f"Assigned {len(dedupe_assignments):,} hashes to {num_groups:,} dedupe groups.")

    # Write assignments
    _write_hash_assignments_batched(hashes_meta, dedupe_assignments, shard_id, total_shards)

    logger.info(
        f"✓ Hash deduplication complete: {len(dedupe_assignments):,} hashes in {num_groups:,} groups."
    )


def _load_and_save_hashes(pb_ids, shard_id, total_shards, cache_file):
    """Load hashes from database and save to HDF5 file"""
    t0 = time.time()
    logger.info("Loading hash metadata from database...")

    hashes_meta = {}
    hash_ids = []
    hash_values = []
    hash_strings = []  # Keep string representation

    # Use JOIN to avoid large IN clause
    query = (
        ImageHash.select(
            ImageHash.id_imagehash,
            ImageHash.pipeline_batch_item,
            ImageHash.detection,
            ImageHash.scan_filename,
            ImageHash.image_hash,
        )
        .join(
            PipelineBatchItem,
            on=(ImageHash.pipeline_batch_item == PipelineBatchItem.id_pipeline_batch_item),
        )
        .where(PipelineBatchItem.pipeline_batch.in_(pb_ids))
    )

    # Add shard filter if sharding - use SQL() to inject raw modulo operation
    if shard_id is not None and total_shards > 1:
        # Use raw SQL to avoid peewee interpreting % as LIKE
        query = query.where(SQL(f"(image_hash.id_imagehash % {total_shards}) = {shard_id}"))

    # Check if query will return any results
    count = query.count()
    logger.info(f"Query will return {count:,} hashes")

    if count == 0:
        logger.warning("No hashes found for this shard")
        return {}, [], np.array([], dtype=object)

    logger.info("Executing query and loading data...")
    hash_length = None
    for i, h in enumerate(query.dicts(), 1):
        hash_id = h["id_imagehash"]
        hex_str = h["image_hash"]

        # Check hash length on first item
        if hash_length is None:
            hash_length = len(hex_str)
            logger.info(f"Hash length: {hash_length} hex chars ({hash_length * 4} bits)")
            if hash_length > 16:
                logger.warning(
                    f"Hash length ({hash_length}) exceeds uint64 capacity (16 hex chars)"
                )
                logger.info("Will use Python arbitrary precision integers")

        try:
            hash_int = int(hex_str, 16)
        except ValueError:
            logger.warning(f"Invalid hash format for ID {hash_id}: {hex_str}")
            hash_int = 0

        hash_ids.append(hash_id)
        hash_values.append(hash_int)
        hash_strings.append(hex_str)
        hashes_meta[hash_id] = {
            "pipeline_batch_item": h["pipeline_batch_item"],
            "detection": h["detection"],
            "scan_filename": h["scan_filename"],
            "image_hash": hex_str,
        }

        # Progress logging for large datasets
        if i % 10000 == 0:
            logger.info(f"Loaded {i:,}/{count:,} hashes ({100*i/count:.1f}%)")

    logger.info(f"Loaded {len(hash_ids):,} hashes in {time.time() - t0:.1f}s")

    # Use object dtype for arbitrary precision Python integers
    # This is needed for hashes longer than 64 bits
    hash_values_array = np.array(hash_values, dtype=object)

    # Save to HDF5 file
    logger.info(f"Saving hashes to {cache_file}...")
    with h5py.File(cache_file, "w") as f:
        f.create_dataset("hash_ids", data=np.array(hash_ids, dtype=np.int64))
        # Store as strings since HDF5 doesn't support arbitrary precision ints
        f.create_dataset(
            "hash_strings", data=np.array(hash_strings, dtype=h5py.string_dtype(encoding="utf-8"))
        )

        # Save metadata as attributes (more efficient for small data)
        meta_group = f.create_group("metadata")
        for hash_id, meta in hashes_meta.items():
            item_group = meta_group.create_group(str(hash_id))
            item_group.attrs["pipeline_batch_item"] = meta["pipeline_batch_item"]
            item_group.attrs["detection"] = meta["detection"]
            item_group.attrs["scan_filename"] = meta["scan_filename"]
            item_group.attrs["image_hash"] = meta["image_hash"]

    logger.info(f"Saved hash data to {cache_file}")

    return hashes_meta, hash_ids, hash_values_array


def _load_hashes_from_file(cache_file):
    """Load hashes from HDF5 file"""
    t0 = time.time()

    with h5py.File(cache_file, "r") as f:
        hash_ids = f["hash_ids"][:].tolist()

        # Load hash strings and convert to integers
        hash_strings = f["hash_strings"][:].astype(str)
        hash_values_array = np.array([int(h, 16) for h in hash_strings], dtype=object)

        # Load metadata
        hashes_meta = {}
        meta_group = f["metadata"]
        for hash_id_str in meta_group.keys():
            hash_id = int(hash_id_str)
            item_group = meta_group[hash_id_str]
            hashes_meta[hash_id] = {
                "pipeline_batch_item": item_group.attrs["pipeline_batch_item"],
                "detection": item_group.attrs["detection"],
                "scan_filename": item_group.attrs["scan_filename"],
                "image_hash": item_group.attrs["image_hash"],
            }

    logger.info(f"Loaded {len(hash_ids):,} hashes from file in {time.time() - t0:.1f}s")

    return hashes_meta, hash_ids, hash_values_array


def _compare_hash_chunk(args):
    """Worker function to compare a chunk of hashes (for multiprocessing)"""
    start_idx, end_idx, hash_ids, hash_values, hamming_threshold, total_hashes = args

    pairs = []

    for i in range(start_idx, end_idx):
        hash_val_i = hash_values[i]
        hash_id_i = hash_ids[i]

        # Compare with all subsequent hashes
        for j in range(i + 1, total_hashes):
            hash_val_j = hash_values[j]
            hash_id_j = hash_ids[j]

            # Calculate Hamming distance using XOR and popcount
            # Python's bin() works with arbitrary precision integers
            xor_result = hash_val_i ^ hash_val_j
            distance = bin(xor_result).count("1")

            if distance <= hamming_threshold:
                pair = tuple(sorted([hash_id_i, hash_id_j]))
                pairs.append(pair)

    return pairs


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


def _find_fuzzy_matches_parallel(hash_ids, hash_values_array, hamming_threshold, workers):
    """Find fuzzy matches using Hamming distance with parallel processing"""
    logger.info(f"Finding fuzzy matches (Hamming distance <= {hamming_threshold})...")
    logger.info(f"Using {workers} parallel workers")

    total_hashes = len(hash_ids)
    logger.info(f"Comparing {total_hashes:,} hashes...")

    # Split work into chunks for parallel processing
    # Each worker gets a range of starting indices
    chunk_size = max(
        1, total_hashes // (workers * 4)
    )  # More chunks than workers for better load balancing
    chunks = []

    for start_idx in range(0, total_hashes, chunk_size):
        end_idx = min(start_idx + chunk_size, total_hashes)
        chunks.append(
            (start_idx, end_idx, hash_ids, hash_values_array, hamming_threshold, total_hashes)
        )

    logger.info(f"Split into {len(chunks)} chunks for parallel processing")

    # Process chunks in parallel
    t0 = time.time()
    similar_pairs = []

    with mp.Pool(processes=workers) as pool:
        results = []
        for i, chunk_pairs in enumerate(pool.imap_unordered(_compare_hash_chunk, chunks), 1):
            similar_pairs.extend(chunk_pairs)
            if i % max(1, len(chunks) // 20) == 0:
                progress = 100 * i / len(chunks)
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - i) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {i}/{len(chunks)} chunks ({progress:.1f}%) - "
                    f"Found {len(similar_pairs):,} pairs - "
                    f"ETA: {eta/60:.1f} min"
                )

    elapsed = time.time() - t0
    logger.info(f"Found {len(similar_pairs):,} fuzzy match pairs in {elapsed:.1f}s")

    # Remove duplicates (pairs might be found by multiple workers)
    similar_pairs = list(set(similar_pairs))
    logger.info(f"After deduplication: {len(similar_pairs):,} unique pairs")

    return similar_pairs


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


def _write_hash_assignments_batched(hashes_meta, dedupe_assignments, shard_id, total_shards):
    """Write deduplication assignments to database in batches"""
    db = DedupedHash._meta.database
    db.create_tables([DedupedHash], safe=True)

    # When sharding, only delete assignments for this shard's hashes
    logger.info("Clearing old assignments for this shard...")
    hash_ids_to_delete = list(dedupe_assignments.keys())

    if hash_ids_to_delete:
        # Delete in chunks to avoid too large IN clause
        chunk_size = 1000
        total_deleted = 0
        for i in range(0, len(hash_ids_to_delete), chunk_size):
            chunk = hash_ids_to_delete[i : i + chunk_size]
            deleted = DedupedHash.delete().where(DedupedHash.hash_id.in_(chunk)).execute()
            total_deleted += deleted
        logger.info(f"Deleted {total_deleted} existing assignments")
    else:
        logger.info("No existing assignments to delete")

    logger.info("Writing new assignments...")
    now = datetime.utcnow()

    chunk_size = HASH_DB_CHUNK_SIZE
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
