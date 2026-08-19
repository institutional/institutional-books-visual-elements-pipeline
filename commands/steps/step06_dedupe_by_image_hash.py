import click
from loguru import logger
import os
import json
import time
import subprocess
import tempfile
import mmap
import array
import ctypes
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import multiprocessing as mp

mp.set_start_method("fork", force=True)

import numpy as np

from models import ImageHash, PipelineBatch, PipelineBatchItem, DedupedHash
from utils import get_time
from const import (
    HASH_DEDUPE_CPUS_LIMIT,
    HASH_DB_CHUNK_SIZE,
    HASH_DEDUPE_HAMMING_THRESHOLD,
    HASH_DEDUPE_CACHE_DIR,
    HASH_DEDUPE_BANDS,
    HASH_DEDUPE_BITS_PER_BAND,
    HASH_DEDUPE_MASK_24,
)

# Globals shared by workers after fork()
_parent_mmap = None
_parent_arr = None


@click.command("step06-dedupe-fast")
@click.option("--id-pipeline-run", type=int, required=True)
@click.option("--hamming-threshold", type=int, default=HASH_DEDUPE_HAMMING_THRESHOLD)
@click.option("--workers", type=int, default=HASH_DEDUPE_CPUS_LIMIT)
def step06_dedupe_by_image_hash(id_pipeline_run, hamming_threshold, workers):
    """
    Deduplicate image hashes using external-sort LSH with mmap bucket processing.

    Uses a fixed LSH band structure (6 bands x 24 bits for 144-bit perceptual hashes)
    to identify candidate pairs, then verifies matches using Hamming distance.

    Processing pipeline:
    1. Loads all hashes from DB into binary files (hashes.bin, hash_ids.bin, metadata.jsonl)
    2. Generates band entries (TSV) for all hashes across 6 LSH bands
    3. External-sorts the band file using GNU sort
    4. Streams sorted buckets, filters oversized buckets (>20k entries)
    5. Processes candidate pairs in parallel using ProcessPoolExecutor with fork start method
    6. Workers read hash data from mmap shared memory (zero-copy via fork inheritance)
    7. Builds Union-Find clusters and writes dedupe assignments to DB in parallel

    NOTE:
    - This command is intended to be run by the orchestrator. See orchestration/execute.py for details.
    - This is a run-level step, which expects a pipeline_run rather than a pipeline_batch.
    - Requires GNU sort for the external sort step.
    """

    logger.info(f"Starting FAST dedupe for run={id_pipeline_run}")
    cache_root = Path(HASH_DEDUPE_CACHE_DIR, f"run_{id_pipeline_run}")
    cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Load hashes from DB
    pb_ids = [
        pb.id_pipeline_batch
        for pb in PipelineBatch.select().where(PipelineBatch.pipeline_run == id_pipeline_run)
    ]
    if not pb_ids:
        logger.error("No pipeline batches found")
        return

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

    total = query.count()
    logger.info(f"Found {total:,} hashes in DB")
    if total == 0:
        return

    logger.info("Loading DB hashes + writing binary files...")

    hashes_bin = cache_root / "hashes.bin"
    hash_ids_bin = cache_root / "hash_ids.bin"
    meta_jsonl = cache_root / "metadata.jsonl"

    hashes_arr = array.array("Q")  # store 144-bit as 3 × 64-bit, though only 144 bits used
    ids_arr = array.array("Q")

    with open(meta_jsonl, "w") as meta_f:
        for row in query.dicts():
            hid = row["id_imagehash"]
            hex_str = row["image_hash"]

            try:
                h_int = int(hex_str, 16)
            except:
                logger.warning(f"Bad hash: {hex_str}")
                h_int = 0

            # For 144 bits we need 3 × 64-bit words
            w0 = h_int & ((1 << 64) - 1)
            w1 = (h_int >> 64) & ((1 << 64) - 1)
            w2 = (h_int >> 128) & ((1 << 64) - 1)

            hashes_arr.append(w0)
            hashes_arr.append(w1)
            hashes_arr.append(w2)

            ids_arr.append(hid)

            meta_f.write(
                json.dumps(
                    {
                        "hash_id": hid,
                        "pipeline_batch_item": row["pipeline_batch_item"],
                        "detection": row["detection"],
                        "image_hash": row["image_hash"],
                    }
                )
                + "\n"
            )

    with open(hashes_bin, "wb") as f:
        hashes_arr.tofile(f)
    with open(hash_ids_bin, "wb") as f:
        ids_arr.tofile(f)

    logger.info("DB load complete.")

    #
    # 2. Write band TSV
    #
    bands_unsorted = cache_root / "bands_unsorted.tsv"
    logger.info("Writing band entries (TSV)...")

    def bands_of(h_int):
        # 144 bits, 6 bands each 24 bits
        for i in range(HASH_DEDUPE_BANDS):
            shift = i * HASH_DEDUPE_BITS_PER_BAND
            yield i, (h_int >> shift) & HASH_DEDUPE_MASK_24

    with open(bands_unsorted, "w") as f:
        for idx in range(total):
            # reconstruct 144-bit integer
            base = idx * 3
            w0 = hashes_arr[base]
            w1 = hashes_arr[base + 1]
            w2 = hashes_arr[base + 2]
            h_int = w0 | (w1 << 64) | (w2 << 128)

            for band_idx, band_val in bands_of(h_int):
                f.write(f"{band_idx}\t{band_val}\t{idx}\n")

    #
    # Next parts will handle: external sort, streaming buckets, workers, clusters, DB writeback
    #
    logger.info("Band TSV complete.")
    run_external_sort_and_process(cache_root, total, hamming_threshold, workers, ids_arr)


def run_external_sort_and_process(cache_root, total, hamming_threshold, workers, ids_arr):
    bands_unsorted = cache_root / "bands_unsorted.tsv"
    bands_sorted = cache_root / "bands_sorted.tsv"

    logger.info("Sorting band file (external GNU sort)...")

    cmd = [
        "sort",
        "-t",
        "\t",
        "-k1,1n",
        "-k2,2n",
        "-o",
        str(bands_sorted),
        str(bands_unsorted),
    ]
    logger.info(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("sort failed: " + result.stderr)

    logger.info("Sort complete. Streaming buckets...")

    def stream_buckets():
        current_key = None
        docs = []

        with open(bands_sorted) as f:
            for line in f:
                band_idx, band_val, didx = line.rstrip("\n").split("\t")
                band_idx = int(band_idx)
                band_val = int(band_val)
                didx = int(didx)

                key = (band_idx, band_val)
                if key != current_key:
                    if current_key is not None and len(docs) > 1:
                        yield docs
                    current_key = key
                    docs = [didx]
                else:
                    docs.append(didx)

        if current_key is not None and len(docs) > 1:
            yield docs

    #
    # Gather buckets (filter size)
    #
    buckets = []
    max_bucket = 20000  # adjustable
    skipped = 0

    for b in stream_buckets():
        if len(b) > max_bucket:
            skipped += 1
            continue
        buckets.append(b)

    logger.info(f"Total buckets: {len(buckets):,} (skipped {skipped:,})")

    build_clusters_and_write(cache_root, total, buckets, hamming_threshold, workers, ids_arr)


#
# Shared mmap buffers for workers
#
_worker_mmap = None
_worker_arr = None  # ctypes array for 64-bit words


def _worker_init():
    # Workers inherit _parent_mmap and _parent_arr from fork()
    global _worker_arr
    _worker_arr = _parent_arr


def _process_chunk(buckets, threshold):
    global _worker_arr
    res = []

    for bucket in buckets:
        docs = sorted(bucket)
        n = len(docs)
        for i in range(n):
            d1 = docs[i]
            base1 = d1 * 3
            h1 = (
                int(_worker_arr[base1])
                | (int(_worker_arr[base1 + 1]) << 64)
                | (int(_worker_arr[base1 + 2]) << 128)
            )
            for j in range(i + 1, n):
                d2 = docs[j]
                base2 = d2 * 3
                h2 = (
                    int(_worker_arr[base2])
                    | (int(_worker_arr[base2 + 1]) << 64)
                    | (int(_worker_arr[base2 + 2]) << 128)
                )
                if bin(h1 ^ h2).count("1") <= threshold:
                    res.append((d1, d2))

    return res


def build_clusters_and_write(cache_root, total, buckets, threshold, workers, ids_arr):
    global _parent_mmap, _parent_arr
    hashes_bin = cache_root / "hashes.bin"
    logger.info("Starting worker pool...")

    #
    # Parallel bucket processing
    #
    seen_pairs = set()
    from utils.unionfind import UnionFindInt

    uf = UnionFindInt(total)

    max_pending = workers * 2
    bucket_iter = iter(buckets)
    pending = set()

    hashes_bin = cache_root / "hashes.bin"

    # Preload mmap in parent so workers inherit it via fork()
    logger.info("Preloading mmap of hashes.bin in parent...")
    with open(hashes_bin, "r+b") as f:
        _parent_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)

    # Build a ctypes array view over the mmap
    total_u64 = total * 3
    _arr_type = ctypes.c_uint64 * total_u64

    _parent_arr = _arr_type.from_buffer(_parent_mmap)
    logger.info("Parent mmap ready (workers will inherit it).")

    # Chunk the buckets BEFORE creating workers
    logger.info("Chunking buckets before processing...")

    # Aim: ~2000–6000 docs per chunk on average
    # Adjust depending on bucket distribution
    CHUNK_SIZE = 5000

    bucket_chunks = []
    current = []
    count_docs = 0

    for b in buckets:
        current.append(b)
        count_docs += len(b)
        if count_docs >= CHUNK_SIZE:
            bucket_chunks.append(current)
            current = []
            count_docs = 0

    # Add final chunk
    if current:
        bucket_chunks.append(current)

    logger.info(f"Total chunks: {len(bucket_chunks):,}")

    logger.info("Starting worker pool with chunked tasks...")

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:

        pending = set()
        chunk_iter = iter(bucket_chunks)

        import itertools

        # initial fill
        for ch in itertools.islice(chunk_iter, max_pending):
            pending.add(executor.submit(_process_chunk, ch, threshold))

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pairs = fut.result()
                for a, b in pairs:
                    if a < b and (a, b) not in seen_pairs:
                        seen_pairs.add((a, b))
                        uf.union(a, b)

            # refill
            for ch in itertools.islice(chunk_iter, len(done)):
                pending.add(executor.submit(_process_chunk, ch, threshold))

    logger.info(f"Verified pairs: {len(seen_pairs):,}")
    logger.info("Building clusters...")

    clusters = uf.get_clusters()

    #
    # Now convert union-find: cluster index → DB hash IDs
    #
    final_assign = {}
    for root, members in clusters.items():
        if len(members) < 1:
            continue
        # representative is min hash ID (like your script)
        rep = min(ids_arr[m] for m in members)
        for m in members:
            final_assign[int(ids_arr[m])] = rep

    logger.info(f"Total deduped groups: {len(clusters):,}")
    write_dedupe_assignments(final_assign, cache_root)


_db_reconnected = False


def _ensure_db_connection():
    """Close inherited connection from fork and create fresh one (once per worker)."""
    global _db_reconnected
    from models import DedupedHash

    db = DedupedHash._meta.database
    if not _db_reconnected:
        if not db.is_closed():
            db.close()
        db.connect()
        _db_reconnected = True
    return db


def _delete_chunk(hash_ids_chunk):
    """Worker function to delete a chunk of old assignments."""
    from models import DedupedHash

    db = _ensure_db_connection()
    with db.atomic():
        DedupedHash.delete().where(DedupedHash.hash_id.in_(hash_ids_chunk)).execute()
    return len(hash_ids_chunk)


def _insert_chunk(rows):
    """Worker function to insert a chunk of rows."""
    from models import DedupedHash

    db = _ensure_db_connection()
    with db.atomic():
        DedupedHash.insert_many(rows).execute()
    return len(rows)


def write_dedupe_assignments(assignments, cache_root, workers=64):
    logger.info("Writing dedupe results to DB...")
    db = DedupedHash._meta.database
    db.create_tables([DedupedHash], safe=True)

    now = get_time()
    hash_ids = list(assignments.keys())
    total = len(hash_ids)

    # Load metadata.jsonl first
    logger.info("Loading metadata...")
    meta_index = {}
    meta_file = cache_root / "metadata.jsonl"
    with open(meta_file) as f:
        for line in f:
            d = json.loads(line)
            hid = d["hash_id"]
            meta_index[hid] = d

    # Prepare all row data upfront
    logger.info("Preparing row data...")
    all_rows = []
    for hid in hash_ids:
        rep = assignments[hid]
        meta = meta_index[hid]
        all_rows.append(
            {
                "hash_id": hid,
                "group_id": rep,
                "pipeline_batch_item": meta["pipeline_batch_item"],
                "detection": meta["detection"],
                "image_hash": meta["image_hash"],
                "created": now,
            }
        )

    # Chunk the data
    chunk_size = HASH_DB_CHUNK_SIZE
    delete_chunks = [hash_ids[i : i + chunk_size] for i in range(0, total, chunk_size)]
    insert_chunks = [all_rows[i : i + chunk_size] for i in range(0, total, chunk_size)]

    logger.info(
        f"Deleting old assignments in parallel ({len(delete_chunks)} chunks, {workers} workers)..."
    )
    deleted = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for count in executor.map(_delete_chunk, delete_chunks):
            deleted += count
            if deleted % 100000 == 0:
                logger.info(f"Deleted {deleted:,}/{total:,}")

    logger.info("Old assignments removed. Writing new ones in parallel...")
    written = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for count in executor.map(_insert_chunk, insert_chunks):
            written += count
            if written % 100000 == 0:
                logger.info(f"Wrote {written:,}/{total:,}")

    logger.info(f"✓ Dedupe complete. Wrote {written:,} rows.")
