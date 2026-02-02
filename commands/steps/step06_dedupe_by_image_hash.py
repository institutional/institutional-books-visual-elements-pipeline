import click
from playhouse.postgres_ext import *
from loguru import logger
import os
import h5py
import numpy as np
import time
from collections import defaultdict
import multiprocessing as mp
import random

from models import ImageHash, PipelineBatch, PipelineBatchItem, DedupedHash
from const import (
    CPUS_LIMIT,
    HASH_DB_CHUNK_SIZE,
    HASH_DEDUPE_HAMMING_THRESHOLD,
    HASH_DEDUPE_CACHE_DIR,
    HASH_DEDUPE_LSH_KEY_SIZE,
    HASH_DEDUPE_LSH_NUM_TABLES,
)
from utils import get_time


@click.command("step06-dedupe-by-image-hash")
@click.option("--id-pipeline-run", type=int, required=True, help="Pipeline run to deduplicate")
@click.option(
    "--hamming-threshold",
    type=int,
    default=HASH_DEDUPE_HAMMING_THRESHOLD,
    show_default=True,
    help="Hamming distance threshold for fuzzy matching (0 = exact match only)",
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
    default=HASH_DEDUPE_CACHE_DIR,
    help="Directory to cache hash data files",
)
@click.option(
    "--force-reload",
    is_flag=True,
    default=False,
    help="Force reload hashes from database (ignore cache)",
)
@click.option(
    "--lsh-num-tables",
    type=int,
    default=HASH_DEDUPE_LSH_NUM_TABLES,
    show_default=True,
    help="Number of LSH hash tables (more = better recall, slower)",
)
@click.option(
    "--lsh-key-size",
    type=int,
    default=HASH_DEDUPE_LSH_KEY_SIZE,
    show_default=True,
    help="Number of bits per LSH key (smaller = more candidates, slower)",
)
def step06_dedupe_by_image_hash(
    id_pipeline_run,
    hamming_threshold,
    workers,
    cache_dir,
    force_reload,
    lsh_num_tables,
    lsh_key_size,
):
    """
    Deduplicate image hashes using exact or fuzzy matching.

    For exact matching (--hamming-threshold=0), uses hash grouping (very fast).
    For fuzzy matching (--hamming-threshold>0), uses LSH for efficient approximate matching.

    Tuning:
    For high recall (find more matches): Increase --lsh-num-tables, decrease --lsh-key-size
    For speed: Decrease --lsh-num-tables, increase --lsh-key-size
    For 256-bit hashes with threshold ≤5: Default settings work well
    For larger thresholds: Increase --lsh-num-tables (each doubling of threshold needs ~2× tables)
    """

    logger.info("Starting hash-based deduplication...")
    logger.info(f"Hamming threshold: {hamming_threshold}")

    if hamming_threshold > 0:
        logger.info(f"LSH configuration: {lsh_num_tables} tables, {lsh_key_size} bits per key")

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

    # Count total hashes
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

    # Generate cache filename
    cache_file = os.path.join(cache_dir, f"hashes_run{id_pipeline_run}.h5")

    # Load or create hash data file
    if os.path.exists(cache_file) and not force_reload:
        logger.info(f"Loading hashes from cache: {cache_file}")
        hashes_meta, hash_ids, hash_values_array = load_hashes_from_file(cache_file)
    else:
        logger.info(f"Loading hashes from database and saving to: {cache_file}")
        hashes_meta, hash_ids, hash_values_array = load_and_save_hashes(pb_ids, cache_file)

    logger.info(f"Loaded {len(hash_ids):,} hashes")

    if not hash_ids:
        logger.warning("No hashes found to deduplicate.")
        return

    # Find similar pairs
    if hamming_threshold == 0:
        similar_pairs = find_exact_matches(hashes_meta)
    else:
        similar_pairs = find_fuzzy_matches_lsh(
            hash_ids, hash_values_array, hamming_threshold, lsh_num_tables, lsh_key_size, workers
        )

    logger.info(f"Found {len(similar_pairs):,} similar pairs")

    # Cluster using Union-Find
    logger.info("Clustering hashes using Union-Find...")
    dedupe_assignments, num_groups = cluster_hashes(hash_ids, similar_pairs)
    logger.info(f"Assigned {len(dedupe_assignments):,} hashes to {num_groups:,} dedupe groups.")

    # Write assignments
    write_hash_assignments_batched(hashes_meta, dedupe_assignments)

    logger.info(
        f"✓ Hash deduplication complete: {len(dedupe_assignments):,} hashes in {num_groups:,} groups."
    )


def load_and_save_hashes(pb_ids, cache_file):
    """Load hashes from database and save to HDF5 file"""
    t0 = time.time()
    logger.info("Loading hash metadata from database...")

    hashes_meta = {}
    hash_ids = []
    hash_values = []
    hash_strings = []

    # Use JOIN to avoid large IN clause
    query = (
        ImageHash.select(
            ImageHash.id_imagehash,
            ImageHash.pipeline_batch_item,
            ImageHash.detection,
            ImageHash.image_hash,
        )
        .join(
            PipelineBatchItem,
            on=(ImageHash.pipeline_batch_item == PipelineBatchItem.id_pipeline_batch_item),
        )
        .where(PipelineBatchItem.pipeline_batch.in_(pb_ids))
    )

    count = query.count()
    logger.info(f"Query will return {count:,} hashes")

    if count == 0:
        logger.warning("No hashes found")
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
            "image_hash": hex_str,
        }

        if i % 10000 == 0:
            logger.info(f"Loaded {i:,}/{count:,} hashes ({100*i/count:.1f}%)")

    logger.info(f"Loaded {len(hash_ids):,} hashes in {time.time() - t0:.1f}s")

    # Use object dtype for arbitrary precision Python integers
    hash_values_array = np.array(hash_values, dtype=object)

    # Save to HDF5 file
    logger.info(f"Saving hashes to {cache_file}...")
    with h5py.File(cache_file, "w") as f:
        f.create_dataset("hash_ids", data=np.array(hash_ids, dtype=np.int64))
        f.create_dataset(
            "hash_strings", data=np.array(hash_strings, dtype=h5py.string_dtype(encoding="utf-8"))
        )

        # Save metadata
        meta_group = f.create_group("metadata")
        for hash_id, meta in hashes_meta.items():
            item_group = meta_group.create_group(str(hash_id))
            item_group.attrs["pipeline_batch_item"] = meta["pipeline_batch_item"]
            item_group.attrs["detection"] = meta["detection"]
            item_group.attrs["image_hash"] = meta["image_hash"]

    logger.info(f"Saved hash data to {cache_file}")

    return hashes_meta, hash_ids, hash_values_array


def load_hashes_from_file(cache_file):
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
                "image_hash": item_group.attrs["image_hash"],
            }

    logger.info(f"Loaded {len(hash_ids):,} hashes from file in {time.time() - t0:.1f}s")

    return hashes_meta, hash_ids, hash_values_array


def find_exact_matches(hashes_meta):
    """Find exact hash matches - very fast O(n)"""
    logger.info("Finding exact hash matches...")

    # Group by hash value
    hash_groups = defaultdict(list)
    for hash_id, meta in hashes_meta.items():
        hash_groups[meta["image_hash"]].append(hash_id)

    # Create pairs from groups
    similar_pairs = []
    duplicates_found = 0

    for hash_value, hash_ids in hash_groups.items():
        if len(hash_ids) > 1:
            duplicates_found += 1
            # Create all pairs within this group
            for i in range(len(hash_ids)):
                for j in range(i + 1, len(hash_ids)):
                    pair = tuple(sorted([hash_ids[i], hash_ids[j]]))
                    similar_pairs.append(pair)

    logger.info(f"Found {len(hash_groups):,} unique hash values")
    logger.info(f"Found {duplicates_found:,} hash values with duplicates")
    logger.info(f"Found {len(similar_pairs):,} exact match pairs")

    return similar_pairs


def find_fuzzy_matches_lsh(
    hash_ids, hash_values_array, hamming_threshold, num_tables, key_size, workers
):
    """
    Find fuzzy matches using LSH (Locality-Sensitive Hashing) for Hamming distance.

    Time complexity: O(n) average case (vs O(n²) brute force)
    Space complexity: O(n * num_tables)

    Args:
        hash_ids: List of hash IDs
        hash_values_array: Array of hash integer values
        hamming_threshold: Maximum Hamming distance to consider as match
        num_tables: Number of LSH hash tables (more = better recall)
        key_size: Bits per LSH key (smaller = more candidates per bucket)
        workers: Number of parallel workers for candidate verification
    """
    logger.info(f"Building LSH index with {num_tables} tables, {key_size} bits per key...")

    # Determine hash bit length from first hash
    if len(hash_values_array) == 0:
        return []

    sample_hash = hash_values_array[0]
    hash_bits = sample_hash.bit_length()
    logger.info(f"Hash size: {hash_bits} bits")

    if key_size >= hash_bits:
        logger.warning(
            f"LSH key_size ({key_size}) >= hash_bits ({hash_bits}), reducing to {hash_bits // 2}"
        )
        key_size = max(1, hash_bits // 2)

    # Create LSH tables with random bit projections
    # Each table samples different random bits from the hash
    lsh_tables = []
    random.seed(42)  # Reproducible

    for table_idx in range(num_tables):
        # Select random bit positions for this table
        bit_positions = random.sample(range(hash_bits), key_size)
        lsh_tables.append({"bit_positions": bit_positions, "buckets": defaultdict(list)})

    # Index all hashes into LSH tables
    logger.info(f"Indexing {len(hash_ids):,} hashes...")
    t0 = time.time()

    for i, hash_id in enumerate(hash_ids):
        hash_val = hash_values_array[i]

        # Add to each LSH table
        for table in lsh_tables:
            # Extract bits at sampled positions to create LSH key
            lsh_key = 0
            for bit_idx, bit_pos in enumerate(table["bit_positions"]):
                if hash_val & (1 << bit_pos):
                    lsh_key |= 1 << bit_idx

            # Add to bucket
            table["buckets"][lsh_key].append(i)  # Store array index, not hash_id

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(hash_ids) - i - 1) / rate
            logger.info(
                f"Indexed {i + 1:,}/{len(hash_ids):,} ({100*(i+1)/len(hash_ids):.1f}%) - "
                f"ETA: {eta:.1f}s"
            )

    logger.info(f"Indexing complete in {time.time() - t0:.1f}s")

    # Log bucket statistics
    for table_idx, table in enumerate(lsh_tables):
        num_buckets = len(table["buckets"])
        bucket_sizes = [len(bucket) for bucket in table["buckets"].values()]
        avg_size = sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0
        max_size = max(bucket_sizes) if bucket_sizes else 0
        logger.info(
            f"Table {table_idx}: {num_buckets:,} buckets, "
            f"avg size: {avg_size:.1f}, max size: {max_size:,}"
        )

    # Find candidate pairs from LSH buckets
    logger.info("Gathering candidate pairs from LSH buckets...")
    candidate_pairs = set()

    for i, hash_id in enumerate(hash_ids):
        hash_val = hash_values_array[i]
        candidates = set()

        # Gather candidates from all tables
        for table in lsh_tables:
            # Get LSH key for this hash
            lsh_key = 0
            for bit_idx, bit_pos in enumerate(table["bit_positions"]):
                if hash_val & (1 << bit_pos):
                    lsh_key |= 1 << bit_idx

            # Get all hashes in same bucket
            bucket = table["buckets"].get(lsh_key, [])
            candidates.update(bucket)

        # Create pairs with candidates (only with higher indices to avoid duplicates)
        for j in candidates:
            if i < j:
                candidate_pairs.add((i, j))

        if (i + 1) % 100000 == 0:
            logger.info(f"Gathered candidates for {i + 1:,}/{len(hash_ids):,} hashes")

    logger.info(f"Found {len(candidate_pairs):,} candidate pairs to verify")

    # Verify candidates in parallel
    logger.info(f"Verifying candidates with {workers} workers...")
    similar_pairs = verify_candidates_parallel(
        candidate_pairs, hash_ids, hash_values_array, hamming_threshold, workers
    )

    logger.info(
        f"Found {len(similar_pairs):,} verified matches within threshold {hamming_threshold}"
    )

    return similar_pairs


def verify_candidates_parallel(candidate_pairs, hash_ids, hash_values_array, threshold, workers):
    """Verify candidate pairs in parallel by computing exact Hamming distance"""
    candidate_list = list(candidate_pairs)

    if not candidate_list:
        return []

    # Split into chunks for parallel processing
    chunk_size = max(1, len(candidate_list) // max(1, workers * 4))
    chunks = [
        (candidate_list[i : i + chunk_size], hash_ids, hash_values_array, threshold)
        for i in range(0, len(candidate_list), chunk_size)
    ]

    logger.info(f"Verifying {len(candidate_list):,} candidates in {len(chunks)} chunks...")

    t0 = time.time()
    verified_pairs = []

    # For large candidate sets this is a CPU-bound operation (bitwise XOR + popcount)
    # and benefits from true parallelism across cores. A process pool helps
    # here because:
    #   - The per-element work is non-trivial compared to the cost of IPC.
    #   - We avoid the GIL when doing the CPU-heavy loop in separate processes.
    # However, for small workloads, process startup/IPC overhead can dominate, so we
    # fall back to a simple in-process loop when either workers <= 1 or we have
    # very few chunks.

    if workers <= 1 or len(chunks) == 1:
        logger.info("Using single-process verification (workers<=1 or small workload).")
        for i, args in enumerate(chunks, 1):
            chunk_pairs = verify_chunk(args)
            verified_pairs.extend(chunk_pairs)

            if i % max(1, len(chunks) // 20) == 0:
                progress = 100 * i / len(chunks)
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(chunks) - i) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {i}/{len(chunks)} chunks ({progress:.1f}%) - "
                    f"Verified {len(verified_pairs):,} matches - "
                    f"ETA: {eta:.1f}s"
                )
    else:
        with mp.Pool(processes=workers) as pool:
            for i, chunk_pairs in enumerate(pool.imap_unordered(verify_chunk, chunks), 1):
                verified_pairs.extend(chunk_pairs)

                if i % max(1, len(chunks) // 20) == 0:
                    progress = 100 * i / len(chunks)
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(chunks) - i) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {i}/{len(chunks)} chunks ({progress:.1f}%) - "
                        f"Verified {len(verified_pairs):,} matches - "
                        f"ETA: {eta:.1f}s"
                    )

    elapsed = time.time() - t0
    logger.info(f"Verification complete in {elapsed:.1f}s")

    return verified_pairs


def verify_chunk(args):
    """Worker function to verify a chunk of candidate pairs"""
    candidate_chunk, hash_ids, hash_values_array, threshold = args

    verified = []

    for i, j in candidate_chunk:
        hash_val_i = hash_values_array[i]
        hash_val_j = hash_values_array[j]

        # Calculate exact Hamming distance
        xor_result = hash_val_i ^ hash_val_j
        distance = bin(xor_result).count("1")

        if distance <= threshold:
            # Store as hash IDs, not array indices
            pair = tuple(sorted([hash_ids[i], hash_ids[j]]))
            verified.append(pair)

    return verified


def cluster_hashes(hash_ids, similar_pairs):
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


def write_hash_assignments_batched(hashes_meta, dedupe_assignments):
    """Write deduplication assignments to database in batches"""
    db = DedupedHash._meta.database
    db.create_tables([DedupedHash], safe=True)

    logger.info("Clearing old assignments...")
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

    logger.info("Writing new assignments...")
    now = get_time()

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
