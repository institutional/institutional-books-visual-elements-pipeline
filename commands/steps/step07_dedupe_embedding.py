import click
from loguru import logger
from models import ImageEmbedding, PipelineBatch, PipelineBatchItem, DedupedEmbedding
import psutil
import faiss
import tqdm
import numpy as np
import time
import os
import h5py

from const import (
    CPUS_LIMIT,
    DEDUPE_EMBEDDING_THRESHOLD,
    DEDUPE_EMBEDDING_MAX_NEIGHBORS,
    DEDUPE_EMBEDDING_MAX_CONNECTIONS,
    DEDUPE_EMBEDDING_HNSW_EF_CONST,
    DEDUPE_EMBEDDING_HNSW_EF_SEARCH,
    DEDUPE_EMBEDDING_HNSW_INDEX_BATCH,
    DEDUPE_EMBEDDING_SEARCH_BATCH,
    DEDUPE_EMBEDDING_CACHE_DIR,
)


@click.command("step07-dedupe-by-image-embeddings")
# TODO: match filename
@click.option("--id-pipeline-run", type=int, required=True, help="Pipeline run to deduplicate")
@click.option(
    "--threshold",
    type=float,
    default=DEDUPE_EMBEDDING_THRESHOLD,
    show_default=True,
    help="Cosine distance threshold for deduplication",
)
@click.option(
    "--max-neighbors",
    type=int,
    default=DEDUPE_EMBEDDING_MAX_NEIGHBORS,
    show_default=True,
    help="Maximum neighbors to find per embedding (k in HNSW search)",
)
@click.option(
    "--hnsw-m",
    type=int,
    default=DEDUPE_EMBEDDING_MAX_CONNECTIONS,
    show_default=True,
    help="HNSW index M parameter (number of connections per layer)",
)
@click.option(
    "--hnsw-ef-construction",
    type=int,
    default=DEDUPE_EMBEDDING_HNSW_EF_CONST,
    show_default=True,
    help="HNSW ef_construction parameter (index build time, higher=better recall)",
)
@click.option(
    "--hnsw-ef-search",
    type=int,
    default=DEDUPE_EMBEDDING_HNSW_EF_SEARCH,
    show_default=True,
    help="HNSW ef_search parameter (query time, higher=better recall)",
)
@click.option(
    "--workers",
    type=int,
    default=CPUS_LIMIT,
    show_default=True,
    help="Number of parallel workers for similarity search",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEDUPE_EMBEDDING_HNSW_INDEX_BATCH,
    show_default=True,
    help="Batch size for index building",
)
@click.option(
    "--search-batch-size",
    type=int,
    default=DEDUPE_EMBEDDING_SEARCH_BATCH,
    show_default=True,
    help="Batch size for similarity search (smaller=less memory)",
)
@click.option(
    "--cache-dir",
    type=str,
    default=DEDUPE_EMBEDDING_CACHE_DIR,
    help="Directory to cache embedding data files",
)
@click.option(
    "--force-reload",
    is_flag=True,
    default=True,
    help="Force reload embeddings from database (ignore cache)",
)
@click.option(
    "--save-index",
    is_flag=True,
    default=False,
    help="Save FAISS index to disk for reuse",
)
def step07_dedupe_embedding(
    id_pipeline_run,
    threshold,
    max_neighbors,
    hnsw_m,
    hnsw_ef_construction,
    hnsw_ef_search,
    workers,
    batch_size,
    search_batch_size,
    cache_dir,
    force_reload,
    save_index,
):
    """
    Deduplicate embeddings at scale using HNSW index with disk caching.

    NOTE: Embeddings are NOT stored in deduped_embedding table (use JOIN with embedding table).
    """

    logger.info(f"Starting deduplication for pipeline run {id_pipeline_run}")
    logger.info(f"Distance threshold: {threshold:.4f}")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Get all pipeline batches for this run
    pipeline_batches = list(
        PipelineBatch.select().where(PipelineBatch.pipeline_run == id_pipeline_run)
    )

    if not pipeline_batches:
        logger.error(f"No pipeline batches found for run {id_pipeline_run}")
        return

    pb_ids = [pb.id_pipeline_batch for pb in pipeline_batches]
    logger.info(f"Found {len(pb_ids)} pipeline batches")

    # Count total embeddings
    total_embeddings = (
        ImageEmbedding.select()
        .join(
            PipelineBatchItem,
            on=(ImageEmbedding.pipeline_batch_item == PipelineBatchItem.id_pipeline_batch_item),
        )
        .where(PipelineBatchItem.pipeline_batch.in_(pb_ids))
        .count()
    )

    logger.info(f"Found {total_embeddings:,} embeddings to deduplicate")

    if total_embeddings == 0:
        logger.warning("No embeddings found, exiting")
        return

    # Generate cache filename
    cache_file = os.path.join(cache_dir, f"embeddings_run{id_pipeline_run}.h5")
    index_file = os.path.join(cache_dir, f"embeddings_run{id_pipeline_run}.index")

    # Load or create embedding data file
    if os.path.exists(cache_file) and not force_reload:
        logger.info(f"Loading embeddings from cache: {cache_file}")
        embedding_metadata, emb_arr = _load_embeddings_from_file(cache_file)
    else:
        logger.info(f"Loading embeddings from database and saving to: {cache_file}")
        embedding_metadata, emb_arr = _load_and_save_embeddings(pb_ids, cache_file)

    logger.info(f"Loaded {len(embedding_metadata):,} embeddings with shape {emb_arr.shape}")

    # Normalize embeddings for cosine similarity
    logger.info("Normalizing embeddings...")
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    emb_arr = emb_arr / (norms + 1e-8)

    # Run deduplication
    logger.info("Starting deduplication process...")
    duplicate_groups = deduplicate_embeddings_hnsw(
        emb_arr,
        threshold=threshold,
        max_neighbors=max_neighbors,
        hnsw_M=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        workers=workers,
        batch_size=batch_size,
        search_batch_size=search_batch_size,
        index_file=index_file if save_index else None,
        force_rebuild_index=force_reload,
    )

    logger.info(
        f"Deduplication complete: {len(duplicate_groups):,} unique groups from {total_embeddings:,} embeddings"
    )

    # Populate DedupedEmbedding table
    _write_dedupe_assignments(embedding_metadata, duplicate_groups)

    # Log statistics
    group_sizes = [len(group) for group in duplicate_groups]
    logger.info(f"=" * 70)
    logger.info(f"DEDUPLICATION STATISTICS")
    logger.info(f"=" * 70)
    logger.info(f"Total embeddings: {total_embeddings:,}")
    logger.info(f"Unique groups: {len(duplicate_groups):,}")
    logger.info(f"Reduction: {100 * (1 - len(duplicate_groups)/total_embeddings):.1f}%")
    logger.info(f"Average group size: {total_embeddings / len(duplicate_groups):.2f}")
    logger.info(f"Min group size: {min(group_sizes)}")
    logger.info(f"Max group size: {max(group_sizes):,}")
    logger.info(f"Median group size: {np.median(group_sizes):.2f}")
    logger.info(f"Groups with size > 1: {sum(1 for s in group_sizes if s > 1):,}")
    logger.info(f"Singletons: {sum(1 for s in group_sizes if s == 1):,}")
    logger.info(f"=" * 70)

    logger.info("✓ Deduplication complete!")


def _load_and_save_embeddings(pb_ids, cache_file):
    """Load embeddings from database and save to HDF5 file"""
    t0 = time.time()
    logger.info("Loading embeddings from database...")

    # Query embeddings
    query = (
        ImageEmbedding.select(
            ImageEmbedding.id_embedding,
            ImageEmbedding.pipeline_batch_item,
            ImageEmbedding.detection,
            ImageEmbedding.scan_filename,
            ImageEmbedding.embedding,
        )
        .join(
            PipelineBatchItem,
            on=(ImageEmbedding.pipeline_batch_item == PipelineBatchItem.id_pipeline_batch_item),
        )
        .where(PipelineBatchItem.pipeline_batch.in_(pb_ids))
        .order_by(ImageEmbedding.id_embedding)
    )

    total = query.count()
    logger.info(f"Query will return {total:,} embeddings")

    embedding_metadata = []
    embedding_vectors = []

    for i, emb in enumerate(query.iterator(), 1):
        embedding_metadata.append(
            {
                "id_embedding": emb.id_embedding,
                "pipeline_batch_item": emb.pipeline_batch_item_id,
                "detection": emb.detection_id,
                "scan_filename": emb.scan_filename,
            }
        )
        embedding_vectors.append(emb.embedding)

        if i % 10000 == 0:
            logger.info(f"Loaded {i:,}/{total:,} embeddings ({100*i/total:.1f}%)")

    # Convert to numpy array
    emb_arr = np.array(embedding_vectors, dtype=np.float32)
    logger.info(f"Loaded {len(embedding_metadata):,} embeddings in {time.time() - t0:.1f}s")
    logger.info(f"Embedding shape: {emb_arr.shape}")

    # Save to HDF5 file
    logger.info(f"Saving embeddings to {cache_file}...")
    with h5py.File(cache_file, "w") as f:
        # Save embedding vectors (compressed)
        f.create_dataset("embeddings", data=emb_arr, compression="gzip", compression_opts=4)

        # Save metadata as separate datasets
        f.create_dataset(
            "embedding_ids",
            data=np.array([m["id_embedding"] for m in embedding_metadata], dtype=np.int64),
        )
        f.create_dataset(
            "pipeline_batch_items",
            data=np.array([m["pipeline_batch_item"] for m in embedding_metadata], dtype=np.int64),
        )
        f.create_dataset(
            "detections",
            data=np.array([m["detection"] for m in embedding_metadata], dtype=np.int64),
        )
        f.create_dataset(
            "scan_filenames",
            data=np.array(
                [m["scan_filename"] for m in embedding_metadata],
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
        )

    file_size_mb = os.path.getsize(cache_file) / 1e6
    logger.info(f"Saved embedding data to {cache_file} ({file_size_mb:.1f} MB)")

    return embedding_metadata, emb_arr


def _load_embeddings_from_file(cache_file):
    """Load embeddings from HDF5 file"""
    t0 = time.time()

    with h5py.File(cache_file, "r") as f:
        emb_arr = f["embeddings"][:]
        embedding_ids = f["embedding_ids"][:].tolist()
        pipeline_batch_items = f["pipeline_batch_items"][:].tolist()
        detections = f["detections"][:].tolist()
        scan_filenames = f["scan_filenames"][:].astype(str).tolist()

    # Reconstruct metadata
    embedding_metadata = []
    for i in range(len(embedding_ids)):
        embedding_metadata.append(
            {
                "id_embedding": embedding_ids[i],
                "pipeline_batch_item": pipeline_batch_items[i],
                "detection": detections[i],
                "scan_filename": scan_filenames[i],
            }
        )

    logger.info(
        f"Loaded {len(embedding_metadata):,} embeddings from file in {time.time() - t0:.1f}s"
    )

    return embedding_metadata, emb_arr


def deduplicate_embeddings_hnsw(
    embeddings,
    threshold,
    max_neighbors=100,
    hnsw_M=16,
    hnsw_ef_construction=200,
    hnsw_ef_search=100,
    workers=CPUS_LIMIT,
    batch_size=100000,
    search_batch_size=10000,
    index_file=None,
    force_rebuild_index=False,
):
    """
    Deduplicate embeddings using FAISS HNSW with optimized memory usage.
    """

    faiss.omp_set_num_threads(workers)

    # Diagnostics
    logger.info(f"CPU diagnostics:")
    logger.info(f"  os.cpu_count() (logical): {os.cpu_count()}")
    logger.info(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"  CPUS_LIMIT: {CPUS_LIMIT}")
    logger.info(f"  FAISS threads: {faiss.omp_get_max_threads()}")

    p = psutil.Process(os.getpid())
    logger.info(f"  CPU affinity: {len(p.cpu_affinity())} cores")

    N, D = embeddings.shape
    normed = embeddings.astype("float32")

    # Convert distance threshold to similarity threshold
    similarity_threshold = 1.0 - threshold
    logger.info(
        f"Distance threshold: {threshold:.4f} → Similarity threshold: {similarity_threshold:.4f}"
    )

    # Try to load existing index
    if index_file and os.path.exists(index_file) and not force_rebuild_index:
        logger.info(f"Loading existing FAISS index from {index_file}...")
        index = faiss.read_index(index_file)
        logger.info("Index loaded successfully")
    else:
        # Build new index
        logger.info(f"Building HNSW index (M={hnsw_M}, ef_construction={hnsw_ef_construction})...")
        index = faiss.IndexHNSWFlat(D, hnsw_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = hnsw_ef_construction

        t0 = time.time()
        for start_idx in tqdm.tqdm(range(0, N, batch_size), desc="Indexing"):
            end_idx = min(start_idx + batch_size, N)
            index.add(normed[start_idx:end_idx])

        logger.info(f"Index built in {time.time() - t0:.1f}s")

        # Save index if requested
        if index_file:
            logger.info(f"Saving index to {index_file}...")
            faiss.write_index(index, index_file)
            index_size_mb = os.path.getsize(index_file) / 1e6
            logger.info(f"Index saved ({index_size_mb:.1f} MB)")

    # Set search parameters
    index.hnsw.efSearch = hnsw_ef_search

    k = min(max_neighbors, N)
    all_links = []

    logger.info(f"Running similarity search (k={k}, ef_search={hnsw_ef_search})...")
    t0 = time.time()
    total_comparisons = 0

    for start in tqdm.tqdm(range(0, N, search_batch_size), desc="Searching"):
        end = min(start + search_batch_size, N)
        batch = normed[start:end]

        t_search0 = time.time()
        sims, idxs = index.search(batch, k)
        t_search1 = time.time()

        # Collect links for this batch
        batch_links = []
        for row_offset, (sim_vec, idx_vec) in enumerate(zip(sims, idxs)):
            i = start + row_offset

            # Find matches above threshold (excluding self)
            mask = (sim_vec >= similarity_threshold) & (idx_vec != i)
            matched_js = idx_vec[mask]

            # Only keep pairs where i < j to avoid duplicates
            for j in matched_js:
                if i < j:
                    batch_links.append((i, int(j)))

        all_links.extend(batch_links)
        total_comparisons += len(batch) * k

        if (start // search_batch_size) % 10 == 0:
            logger.info(
                f"  Batch {start:,}-{end:,}: {t_search1 - t_search0:.2f}s, "
                f"found {len(batch_links):,} links (total: {len(all_links):,})"
            )

    elapsed = time.time() - t0
    logger.info(
        f"Search complete in {elapsed:.1f}s "
        f"({total_comparisons/elapsed/1e6:.2f}M comparisons/sec)"
    )
    logger.info(f"Collected {len(all_links):,} duplicate links")

    # Group duplicates using Union-Find
    logger.info("Computing duplicate groups with Union-Find...")
    t0 = time.time()
    parent = np.arange(N, dtype=np.int32)

    def find(i):
        """Find with path compression"""
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        # Path compression
        for x in path:
            parent[x] = i
        return i

    def union(i, j):
        """Union by rank (implicit)"""
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Process all links
    for i, (u, v) in enumerate(all_links):
        union(u, v)
        if (i + 1) % 1000000 == 0:
            logger.info(f"  Processed {i + 1:,}/{len(all_links):,} links")

    # Gather groups
    logger.info("Gathering groups...")
    group_map = {}
    for idx in range(N):
        root = find(idx)
        if root not in group_map:
            group_map[root] = []
        group_map[root].append(idx)

    groups = list(group_map.values())
    logger.info(f"Found {len(groups):,} unique groups in {time.time() - t0:.1f}s")

    return groups


def _write_dedupe_assignments(embedding_metadata, duplicate_groups):
    """
    Write deduplication assignments to database WITHOUT storing embeddings.

    Embeddings can be retrieved by joining with the embedding table:
    SELECT de.*, e.embedding
    FROM deduped_embedding de
    JOIN embedding e ON e.id_embedding = de.embedding_id
    """
    db = DedupedEmbedding._meta.database

    logger.info("Preparing dedupe assignments...")
    t0 = time.time()

    # Build all assignments
    assignments = []
    for group_id, group_indices in enumerate(duplicate_groups):
        # Use minimum embedding ID as group representative for consistency
        representative_idx = min(group_indices)
        representative_id = embedding_metadata[representative_idx]["id_embedding"]

        for idx in group_indices:
            metadata = embedding_metadata[idx]
            assignments.append(
                {
                    "embedding_id": metadata["id_embedding"],
                    "group_id": representative_id,  # Use representative's ID as group ID
                    "pipeline_batch_item": metadata["pipeline_batch_item"],
                    "detection": metadata["detection"],
                    "scan_filename": metadata["scan_filename"],
                }
            )

    logger.info(f"Prepared {len(assignments):,} assignments in {time.time() - t0:.1f}s")

    # Delete old assignments for these embeddings
    logger.info("Clearing old assignments...")
    embedding_ids = [a["embedding_id"] for a in assignments]

    delete_chunk_size = 10000
    total_deleted = 0

    for i in range(0, len(embedding_ids), delete_chunk_size):
        chunk = embedding_ids[i : i + delete_chunk_size]
        deleted = (
            DedupedEmbedding.delete().where(DedupedEmbedding.embedding_id.in_(chunk)).execute()
        )
        total_deleted += deleted

    logger.info(f"Deleted {total_deleted:,} existing assignments")

    logger.info("Writing new assignments...")
    insert_batch_size = 10000
    total_written = 0

    with db.atomic():
        for i in tqdm.tqdm(range(0, len(assignments), insert_batch_size), desc="Inserting"):
            batch = assignments[i : i + insert_batch_size]
            DedupedEmbedding.insert_many(batch).execute()
            total_written += len(batch)

    logger.info(
        f"✓ Successfully wrote {total_written:,} deduplicated embeddings in {time.time() - t0:.1f}s"
    )


@click.command("migrate-dedupe-embedding")
def migrate_dedupe_embedding():
    """
    Make embedding column nullable in deduped_embedding table.
    Run this once before using the optimized deduplication.
    """
    db = DedupedEmbedding._meta.database

    logger.info("Running migration: making embedding column nullable...")

    try:
        with db.atomic():
            db.execute_sql("ALTER TABLE deduped_embedding ALTER COLUMN embedding DROP NOT NULL")

        logger.info("✓ Migration complete: embedding column is now nullable")
        logger.info("  You can now use step07-dedupe-embeddings without storing embeddings")
        logger.info("  This saves disk space and improves performance significantly!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.info("If column is already nullable or doesn't exist, this is safe to ignore")
