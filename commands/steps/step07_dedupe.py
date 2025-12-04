import click
from loguru import logger
from models import Embedding, PipelineBatch, PipelineBatchItem, Detection, DedupedEmbedding
import psutil
import faiss
import tqdm
import numpy as np
import time
import os
import h5py

from const import CPUS_LIMIT


@click.command("step07-dedupe-embeddings")
@click.option("--id-pipeline-run", type=int, required=True, help="Pipeline run to deduplicate")
@click.option(
    "--threshold",
    type=float,
    default=0.14,
    show_default=True,
    help="Cosine distance threshold for deduplication",
)
@click.option(
    "--max-neighbors",
    type=int,
    default=100,
    show_default=True,
    help="Maximum neighbors to find per embedding (k in HNSW search)",
)
@click.option(
    "--hnsw-m",
    type=int,
    default=16,
    show_default=True,
    help="HNSW index M parameter (number of connections per layer)",
)
@click.option(
    "--hnsw-ef-construction",
    type=int,
    default=64,
    show_default=True,
    help="HNSW ef_construction parameter (index build time)",
)
@click.option(
    "--hnsw-ef-search",
    type=int,
    default=40,
    show_default=True,
    help="HNSW ef_search parameter (query time)",
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
    default=70000,
    show_default=True,
    help="Batch size for index building",
)
@click.option(
    "--search-batch-size",
    type=int,
    default=70000,
    show_default=True,
    help="Batch size for similarity search",
)
def step07_dedupe(
    id_pipeline_run,
    threshold,
    max_neighbors,
    hnsw_m,
    hnsw_ef_construction,
    hnsw_ef_search,
    workers,
    batch_size,
    search_batch_size,
):
    """
    Deduplicate embeddings at scale using HNSW index.
    """

    logger.info(f"Starting deduplication for pipeline run {id_pipeline_run}")

    # Get all pipeline batches for this run
    pipeline_batches = list(
        PipelineBatch.select().where(PipelineBatch.pipeline_run == id_pipeline_run)
    )

    if not pipeline_batches:
        logger.error(f"No pipeline batches found for run {id_pipeline_run}")
        return

    pb_item_ids = [item.id_pipeline_batch_item for pb in pipeline_batches for item in pb.items]

    logger.info(f"Found {len(pb_item_ids)} pipeline batch items")

    # Load all embeddings from database
    logger.info("Loading embeddings from database...")
    embeddings_query = (
        Embedding.select()
        .where(Embedding.pipeline_batch_item.in_(pb_item_ids))
        .order_by(Embedding.id_embedding)
    )

    total_embeddings = embeddings_query.count()
    logger.info(f"Found {total_embeddings} embeddings to deduplicate")

    if total_embeddings == 0:
        logger.warning("No embeddings found, exiting")
        return

    # Extract embeddings and metadata
    embedding_vectors = []
    embedding_metadata = []

    for emb in tqdm.tqdm(
        embeddings_query.iterator(), total=total_embeddings, desc="Loading embeddings"
    ):
        embedding_vectors.append(emb.embedding)
        embedding_metadata.append(
            {
                "id_embedding": emb.id_embedding,
                "pipeline_batch_item": emb.pipeline_batch_item_id,
                "detection": emb.detection_id,
                "scan_filename": emb.scan_filename,
            }
        )

    # Convert to numpy array and normalize
    emb_arr = np.array(embedding_vectors, dtype=np.float32)
    logger.info(f"Embedding array shape: {emb_arr.shape}")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    emb_arr = emb_arr / (norms + 1e-8)

    # Run deduplication
    logger.info("Starting deduplication process...")
    unique_indices, duplicate_groups = deduplicate_embeddings_hnsw(
        emb_arr,
        threshold=threshold,
        max_neighbors=max_neighbors,
        hnsw_M=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        workers=workers,
        batch_size=batch_size,
        search_batch_size=search_batch_size,
    )

    logger.info(
        f"Deduplication complete: {len(unique_indices)} unique groups from {total_embeddings} embeddings"
    )

    # Populate DedupedEmbedding table
    logger.info("Populating DedupedEmbedding table...")

    # Clear existing deduped embeddings for this pipeline run
    deleted = (
        DedupedEmbedding.delete()
        .where(DedupedEmbedding.pipeline_batch_item.in_(pb_item_ids))
        .execute()
    )
    logger.info(f"Deleted {deleted} existing deduped embeddings")

    # Insert new deduplicated embeddings
    batch_insert_data = []

    for group_id, group_indices in enumerate(
        tqdm.tqdm(duplicate_groups, desc="Preparing insert data")
    ):
        for idx in group_indices:
            metadata = embedding_metadata[idx]
            batch_insert_data.append(
                {
                    "embedding_id": metadata["id_embedding"],
                    "group_id": group_id,
                    "pipeline_batch_item": metadata["pipeline_batch_item"],
                    "detection": metadata["detection"],
                    "scan_filename": metadata["scan_filename"],
                    "embedding": emb_arr[
                        idx
                    ].tolist(),  # Convert numpy array to list for ArrayField
                }
            )

    # Batch insert into database
    logger.info(f"Inserting {len(batch_insert_data)} records into DedupedEmbedding table...")
    insert_batch_size = 1000
    for i in tqdm.tqdm(range(0, len(batch_insert_data), insert_batch_size), desc="Inserting"):
        batch = batch_insert_data[i : i + insert_batch_size]
        DedupedEmbedding.insert_many(batch).execute()

    logger.info("Deduplication complete!")
    logger.info(f"Total embeddings: {total_embeddings}")
    logger.info(f"Unique groups: {len(duplicate_groups)}")
    logger.info(f"Average group size: {total_embeddings / len(duplicate_groups):.2f}")

    # Log some statistics about group sizes
    group_sizes = [len(group) for group in duplicate_groups]
    logger.info(f"Min group size: {min(group_sizes)}")
    logger.info(f"Max group size: {max(group_sizes)}")
    logger.info(f"Median group size: {np.median(group_sizes):.2f}")
    logger.info(f"Groups with size > 1: {sum(1 for s in group_sizes if s > 1)}")


def deduplicate_embeddings_hnsw(
    embeddings,
    threshold,
    max_neighbors=100,
    hnsw_M=16,
    hnsw_ef_construction=64,
    hnsw_ef_search=40,
    workers=CPUS_LIMIT,
    batch_size=70000,
    search_batch_size=70000,
):
    """Deduplicate embeddings using FAISS HNSW in batch with optimized grouping."""

    faiss.omp_set_num_threads(workers)

    # Diagnostics
    print("os.cpu_count() (logical):", os.cpu_count())
    print("CPUS_LIMIT:", CPUS_LIMIT)
    print(
        "psutil.cpu_count(logical=False) (physical cores):",
        psutil.cpu_count(logical=False),
    )
    p = psutil.Process(os.getpid())
    affinity = p.cpu_affinity()
    print("CPU affinity of this process:", affinity)
    print("Num CPUs this process can use:", len(affinity))
    print("faiss.omp_get_max_threads():", faiss.omp_get_max_threads())

    N, D = embeddings.shape
    normed = embeddings.astype("float32")
    index = faiss.IndexHNSWFlat(D, hnsw_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = hnsw_ef_construction
    index.hnsw.efSearch = hnsw_ef_search

    # Convert distance threshold to similarity threshold
    # distance = 1 - similarity (for cosine distance)
    similarity_threshold = 1.0 - threshold
    print(f"Distance threshold: {threshold:.4f} → Similarity threshold: {similarity_threshold:.4f}")

    # Build the index
    print("Building HNSW index...")
    for start_idx in tqdm.tqdm(range(0, N, batch_size), desc="Indexing", unit="vec"):
        end_idx = min(start_idx + batch_size, N)
        index.add(normed[start_idx:end_idx])

    k = min(max_neighbors, N)
    all_links = []  # To store pairs (i, j) where i ~ j

    print("Running batched search and collecting links...")
    time0 = time.time()
    for start in tqdm.tqdm(range(0, N, search_batch_size), desc="Searching", unit="img"):
        end = min(start + search_batch_size, N)
        t_search0 = time.time()
        sims, idxs = index.search(normed[start:end], k)  # shape (batch, k)
        t_search1 = time.time()
        print("FAISS SEARCH BATCH TIME", t_search1 - t_search0, "BATCH SIZE", end - start)

        # For batch [start, end)
        row_indices = np.arange(start, end)
        for row_offset, (sim_vec, idx_vec) in enumerate(zip(sims, idxs)):
            i = row_indices[row_offset]
            # Use similarity_threshold instead of threshold
            mask = (sim_vec >= similarity_threshold) & (idx_vec != i)
            matched_js = idx_vec[mask]
            # Record all pairs (i, j) with sufficient similarity
            all_links.extend([(i, int(j)) for j in matched_js])

    print(f"Collected {len(all_links)} duplicate links in {time.time() - time0:.1f} seconds.")

    # Now, group duplicates using Union-Find (connected components)
    print("Computing duplicate groups...")
    parent = np.arange(N)  # parent[i] = i

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for x in path:
            parent[x] = i
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    for i, j in all_links:
        union(i, j)

    # Gather groups
    group_map = {}
    for idx in range(N):
        root = find(idx)
        group_map.setdefault(root, []).append(idx)

    # Extract unique representatives (first of each group)
    unique_indices = [min(group) for group in group_map.values()]
    groups = list(group_map.values())

    print(f"Found {len(groups)} unique groups from {N} vectors.")

    return np.array(unique_indices), groups
