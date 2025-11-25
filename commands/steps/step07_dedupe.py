import click
from peewee import fn
from playhouse.postgres_ext import *
from loguru import logger
from models import Embedding, PipelineBatch, PipelineBatchItem, Detection, DedupedEmbedding
from datetime import datetime
from itertools import count
import psycopg2.extras
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import os
from const import CPUS_LIMIT
import time


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
    "--search-batch-size",
    type=int,
    default=500,
    show_default=True,
    help="Number of embeddings per worker batch",
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
    "--checkpoint-file",
    type=str,
    default="dedupe_checkpoint.pkl",
    help="File to save progress for resuming",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from checkpoint file",
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
    "--max-embeddings-per-shard",
    type=int,
    default=1_000_000,
    show_default=True,
    help="Maximum embeddings per shard (auto-shard if exceeded)",
)
def step07_dedupe(
    id_pipeline_run,
    threshold,
    search_batch_size,
    max_neighbors,
    hnsw_m,
    hnsw_ef_construction,
    hnsw_ef_search,
    workers,
    checkpoint_file,
    resume,
    shard_id,
    total_shards,
    max_embeddings_per_shard,
):
    """
    Deduplicate embeddings at scale using HNSW index with sharding support.

    For large datasets (>1M embeddings), use sharding:

    \b
    # Auto-shard (sequential):
    python pipeline.py step07-dedupe --id-pipeline-run=1

    \b
    # Manual parallel sharding (run these in parallel):
    python pipeline.py step07-dedupe --id-pipeline-run=1 --shard-id=0 --total-shards=10 &
    python pipeline.py step07-dedupe --id-pipeline-run=1 --shard-id=1 --total-shards=10 &
    ...
    python pipeline.py step07-dedupe --id-pipeline-run=1 --shard-id=9 --total-shards=10 &
    """

    logger.info("Checking database configuration...")
    _check_database_settings()

    # Count total embeddings
    pipeline_batches = list(
        PipelineBatch.select().where(PipelineBatch.pipeline_run == id_pipeline_run)
    )
    pb_item_ids = [item.id_pipeline_batch_item for pb in pipeline_batches for item in pb.items]

    total_embeddings = (
        Embedding.select().where(Embedding.pipeline_batch_item.in_(pb_item_ids)).count()
    )

    logger.info(f"Total embeddings in pipeline run {id_pipeline_run}: {total_embeddings:,}")

    # Auto-shard if needed
    if shard_id is None and total_embeddings > max_embeddings_per_shard:
        required_shards = (total_embeddings // max_embeddings_per_shard) + 1
        logger.info(f"Dataset exceeds {max_embeddings_per_shard:,} embeddings per shard")
        logger.info(f"Auto-sharding into {required_shards} shards")
        logger.info("=" * 70)
        logger.info("For faster processing, run shards in parallel:")
        logger.info("")
        for i in range(required_shards):
            logger.info(
                f"  python pipeline.py step07-dedupe --id-pipeline-run={id_pipeline_run} "
                f"--shard-id={i} --total-shards={required_shards} &"
            )
        logger.info("")
        logger.info("=" * 70)
        logger.info("Processing all shards sequentially...")

        for i in range(required_shards):
            logger.info(f"\n{'='*70}")
            logger.info(f"SHARD {i+1}/{required_shards}")
            logger.info(f"{'='*70}\n")

            # Use shard-specific checkpoint file
            shard_checkpoint = f"dedupe_checkpoint_shard{i}_of_{required_shards}.pkl"

            _process_single_shard(
                id_pipeline_run=id_pipeline_run,
                pb_item_ids=pb_item_ids,
                shard_id=i,
                total_shards=required_shards,
                threshold=threshold,
                search_batch_size=search_batch_size,
                max_neighbors=max_neighbors,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                hnsw_ef_search=hnsw_ef_search,
                workers=workers,
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
        checkpoint_file = f"dedupe_checkpoint_shard{shard_id}_of_{total_shards}.pkl"

    _process_single_shard(
        id_pipeline_run=id_pipeline_run,
        pb_item_ids=pb_item_ids,
        shard_id=shard_id,
        total_shards=total_shards,
        threshold=threshold,
        search_batch_size=search_batch_size,
        max_neighbors=max_neighbors,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        workers=workers,
        checkpoint_file=checkpoint_file,
        resume=resume,
    )


def _process_single_shard(
    id_pipeline_run,
    pb_item_ids,
    shard_id,
    total_shards,
    threshold,
    search_batch_size,
    max_neighbors,
    hnsw_m,
    hnsw_ef_construction,
    hnsw_ef_search,
    workers,
    checkpoint_file,
    resume,
):
    """Process a single shard of embeddings"""

    logger.info(
        f"Starting deduplication for pipeline run {id_pipeline_run} "
        f"(cosine threshold {threshold})"
    )
    logger.info(
        f"HNSW parameters: M={hnsw_m}, ef_construction={hnsw_ef_construction}, ef_search={hnsw_ef_search}"
    )
    logger.info(f"Search batch size: {search_batch_size} embeddings per worker, {workers} workers")

    if shard_id is not None:
        logger.info(f"Shard: {shard_id + 1}/{total_shards}")

    # Get embeddings for this shard
    t0 = time.time()
    logger.info("Loading embedding metadata...")

    embeddings_meta = {}
    query = Embedding.select(
        Embedding.id_embedding,
        Embedding.pipeline_batch_item,
        Embedding.detection,
        Embedding.scan_filename,
    ).where(Embedding.pipeline_batch_item.in_(pb_item_ids))

    # Add shard filter if sharding
    if shard_id is not None and total_shards > 1:
        query = query.where((Embedding.id_embedding % total_shards) == shard_id)

    for e in query.dicts():
        embeddings_meta[e["id_embedding"]] = {
            "pipeline_batch_item": e["pipeline_batch_item"],
            "detection": e["detection"],
            "scan_filename": e["scan_filename"],
        }

    logger.info(f"Found {len(embeddings_meta):,} embeddings to process in {time.time() - t0:.1f}s")

    if not embeddings_meta:
        logger.warning("No embeddings found to deduplicate.")
        return

    # Estimate memory usage
    embedding_ids = list(embeddings_meta.keys())
    estimated_index_size_mb = (
        len(embedding_ids) * 512 * 4 / (1024 * 1024) * 2.6
    )  # Rough HNSW overhead
    logger.info(f"Estimated HNSW index size: ~{estimated_index_size_mb:.0f} MB")

    # Ensure HNSW index exists
    logger.info("Ensuring HNSW index exists...")
    _ensure_hnsw_index(hnsw_m, hnsw_ef_construction)

    # Set ef_search parameter
    _set_hnsw_ef_search(hnsw_ef_search)

    # Find similar pairs
    similar_pairs = _find_similar_pairs_batched(
        embeddings_meta,
        pb_item_ids,
        shard_id,
        total_shards,
        threshold,
        search_batch_size,
        max_neighbors,
        workers,
        checkpoint_file,
        resume,
    )

    logger.info(f"Found {len(similar_pairs):,} unique similar pairs")

    # Cluster using Union-Find
    logger.info("Clustering embeddings using Union-Find...")
    dedupe_assignments, num_groups = _cluster_embeddings(embeddings_meta.keys(), similar_pairs)
    logger.info(f"Assigned {len(dedupe_assignments):,} embeddings to {num_groups:,} dedupe groups.")

    # Write assignments
    _write_assignments_batched(
        embeddings_meta, dedupe_assignments, pb_item_ids, shard_id, total_shards
    )

    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info(f"Removed checkpoint file: {checkpoint_file}")

    logger.info(
        f"✓ Deduplication complete: {len(dedupe_assignments):,} embeddings in {num_groups:,} groups."
    )


def _check_database_settings():
    """Check and optimize database settings for vector search"""
    db = Embedding._meta.database

    # Check pgvector extension
    try:
        result = db.execute_sql(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if result:
            logger.info(f"✓ pgvector extension version: {result[0]}")
        else:
            logger.error("✗ pgvector extension not installed!")
            return False
    except:
        logger.error("✗ Could not check pgvector extension")
        return False

    # Check work_mem
    work_mem = db.execute_sql("SHOW work_mem").fetchone()[0]
    logger.info(f"work_mem: {work_mem} (recommend >= 256MB for vector ops)")

    # Increase work_mem for this session
    try:
        db.execute_sql("SET work_mem = '256MB'")
        new_work_mem = db.execute_sql("SHOW work_mem").fetchone()[0]
        logger.info(f"✓ Increased work_mem to {new_work_mem} for this session")
    except Exception as e:
        logger.warning(f"Could not set work_mem: {e}")

    # Increase maintenance_work_mem for index operations
    try:
        db.execute_sql("SET maintenance_work_mem = '1GB'")
        logger.info("✓ Set maintenance_work_mem to 1GB")
    except Exception as e:
        logger.warning(f"Could not set maintenance_work_mem: {e}")

    # Check shared_buffers
    shared_buffers = db.execute_sql("SHOW shared_buffers").fetchone()[0]
    logger.info(f"shared_buffers: {shared_buffers}")

    # Check max_connections
    max_conn = db.execute_sql("SHOW max_connections").fetchone()[0]
    logger.info(f"max_connections: {max_conn}")

    # Disable JIT for vector operations
    try:
        db.execute_sql("SET jit = off")
        logger.info("✓ Disabled JIT compilation")
    except:
        pass

    return True


def _ensure_hnsw_index(m, ef_construction):
    """Ensure HNSW index exists on embedding column with specified parameters."""
    db = Embedding._meta.database

    # Check existing indexes
    logger.info("Checking existing indexes on embedding table...")
    try:
        result = db.execute_sql(
            """
            SELECT 
                indexname, 
                indexdef,
                pg_size_pretty(pg_relation_size(indexname::regclass)) as size
            FROM pg_indexes 
            WHERE tablename = 'embedding'
        """
        ).fetchall()

        hnsw_found = False
        for idx_name, idx_def, size in result:
            logger.info(f"  Index: {idx_name} ({size})")
            if "hnsw" in idx_def.lower():
                logger.info(f"    Type: HNSW")
                if "embedding" in idx_def.lower() and "vector_cosine_ops" in idx_def.lower():
                    logger.info(f"    ✓ Correct HNSW index on embedding column with cosine ops")
                    hnsw_found = True
            else:
                idx_type = idx_def.split("USING")[1].split()[0] if "USING" in idx_def else "btree"
                logger.info(f"    Type: {idx_type}")

        if hnsw_found:
            logger.info("✓ HNSW index exists and is configured correctly")
            return
        else:
            logger.warning("✗ No HNSW index found on embedding column with vector_cosine_ops!")

    except Exception as e:
        logger.warning(f"Could not check for existing indexes: {e}")

    # Check table size before creating index
    try:
        stats = db.execute_sql(
            """
            SELECT 
                count(*) as rows,
                pg_size_pretty(pg_total_relation_size('embedding')) as size
            FROM embedding
        """
        ).fetchone()
        logger.info(f"Table stats: {stats[0]:,} rows, {stats[1]} total size")
        estimated_minutes = stats[0] / 10000
        logger.info(f"Index creation estimated time: ~{estimated_minutes:.0f} minutes")
    except:
        pass

    # Create HNSW index
    try:
        logger.info(f"Creating HNSW index with M={m}, ef_construction={ef_construction}...")
        logger.info("This will run in background with CREATE INDEX CONCURRENTLY...")

        t0 = time.time()
        db.execute_sql(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS embedding_hnsw_idx 
            ON embedding 
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {m}, ef_construction = {ef_construction})
        """
        )
        elapsed = time.time() - t0
        logger.info(f"✓ HNSW index created successfully in {elapsed:.1f}s")

        # Verify it was created
        result = db.execute_sql(
            """
            SELECT pg_size_pretty(pg_relation_size('embedding_hnsw_idx'))
        """
        ).fetchone()
        if result:
            logger.info(f"  Index size: {result[0]}")

    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg:
            logger.info("HNSW index already exists")
        elif "duplicate key" in error_msg:
            logger.info("HNSW index already exists (duplicate key)")
        else:
            logger.error(f"Could not create HNSW index: {e}")
            logger.error("Vector search will be VERY SLOW without index!")


def _set_hnsw_ef_search(ef_search):
    """Set HNSW ef_search parameter for current session."""
    db = Embedding._meta.database
    try:
        db.execute_sql(f"SET hnsw.ef_search = {ef_search}")
        logger.info(f"✓ Set hnsw.ef_search = {ef_search} for this session")
    except Exception as e:
        logger.warning(f"Could not set hnsw.ef_search: {e}")

    # Test query to verify index usage
    logger.info("Testing index usage with EXPLAIN...")
    try:
        test_emb = Embedding.select().limit(1).first()
        if test_emb:
            vec = test_emb.embedding
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            # Get query plan
            import json

            plan = db.execute_sql(
                """
                EXPLAIN (FORMAT JSON)
                SELECT id_embedding
                FROM embedding
                ORDER BY embedding <=> %s::vector
                LIMIT 100
            """,
                (vec,),
            ).fetchone()[0]

            plan_text = json.dumps(plan, indent=2)

            if "embedding_hnsw_idx" in plan_text:
                logger.info("✓ Query plan confirms HNSW index will be used")
            elif "Index Scan" in plan_text or "Bitmap" in plan_text:
                logger.info("✓ Query will use an index (but may not be HNSW)")
                logger.debug(f"Plan: {plan_text[:500]}")
            else:
                logger.warning("✗ Query plan shows Sequential Scan - index not being used!")
                logger.warning(f"Plan: {plan_text[:500]}")
                logger.warning("This will be EXTREMELY SLOW!")

    except Exception as e:
        logger.warning(f"Could not test query plan: {e}")


def _find_similar_pairs_batched(
    embeddings_meta,
    pb_item_ids,
    shard_id,
    total_shards,
    threshold,
    search_batch_size,
    max_neighbors,
    workers,
    checkpoint_file,
    resume,
):
    """Find similar pairs using parallel workers."""
    embedding_ids = list(embeddings_meta.keys())
    sim_threshold = 1 - threshold  # Convert cosine distance to similarity
    N = len(embedding_ids)

    # Adjust k
    k = min(max_neighbors, N)
    logger.info(f"Will search for top k={k} neighbors per embedding")

    # Load checkpoint if resuming
    processed_batches = set()
    similar_pairs_set = set()

    if resume and os.path.exists(checkpoint_file):
        logger.info(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
            processed_batches = checkpoint["processed_batches"]
            similar_pairs_set = set(
                checkpoint.get("similar_pairs_set", checkpoint.get("similar_pairs", []))
            )
        logger.info(
            f"Resumed: {len(processed_batches)} batches already processed, "
            f"{len(similar_pairs_set)} pairs found"
        )

    # Create batches
    batches = []
    for start_idx in range(0, N, search_batch_size):
        batch_idx = start_idx // search_batch_size
        end_idx = min(start_idx + search_batch_size, N)
        if batch_idx not in processed_batches:
            batches.append((batch_idx, start_idx, end_idx, embedding_ids[start_idx:end_idx]))

    total_batches = (N + search_batch_size - 1) // search_batch_size
    logger.info(
        f"Total batches: {total_batches}, Already processed: {len(processed_batches)}, "
        f"Remaining: {len(batches)}"
    )
    logger.info(f"Each batch processes ~{search_batch_size} embeddings with {k} neighbors each")
    logger.info(f"Using {workers} parallel workers")

    if not batches:
        logger.info("All batches already processed!")
        return list(similar_pairs_set)

    # Test database connection
    logger.info("Testing database connection...")
    try:
        test_count = Embedding.select().where(Embedding.id_embedding.in_(embedding_ids[:1])).count()
        logger.info(f"✓ Database connection OK (test query returned {test_count})")
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
        raise

    # Process batches in parallel
    logger.info(f"Running batched search and collecting links...")
    logger.info(f"Submitting {len(batches)} batches to {workers} workers...")
    time0 = time.time()
    completed_count = len(processed_batches)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all futures
        futures = {}
        for batch_idx, start_idx, end_idx, batch_ids in batches:
            future = executor.submit(
                _process_search_batch,
                batch_idx,
                start_idx,
                end_idx,
                batch_ids,
                pb_item_ids,
                shard_id,
                total_shards,
                sim_threshold,
                k,
            )
            futures[future] = (batch_idx, end_idx - start_idx)

        logger.info(f"✓ Submitted {len(futures)} batches to executor")

        checkpoint_interval = max(1, len(batches) // 20)  # Checkpoint ~20 times
        batches_since_checkpoint = 0
        last_log_time = time.time()

        for idx, future in enumerate(as_completed(futures, timeout=3600), 1):
            batch_idx, batch_size = futures[future]
            try:
                batch_pairs, search_time, queries_per_sec = future.result(timeout=600)

                # Add to set
                pairs_before = len(similar_pairs_set)
                similar_pairs_set.update(batch_pairs)
                new_pairs = len(similar_pairs_set) - pairs_before

                processed_batches.add(batch_idx)
                completed_count += 1
                batches_since_checkpoint += 1

                # Calculate progress
                elapsed = time.time() - time0
                batches_done = idx
                rate = batches_done / elapsed if elapsed > 0 else 0
                remaining = len(batches) - batches_done
                eta = remaining / rate if rate > 0 else 0

                # Log progress
                now = time.time()
                if now - last_log_time >= 5 or idx % max(1, len(batches) // 100) == 0:
                    logger.info(
                        f"Batch {batches_done}/{len(batches)} ({100*batches_done/len(batches):.1f}%) | "
                        f"Batch #{batch_idx}: {batch_size} embs in {search_time:.1f}s ({queries_per_sec:.1f} q/s) | "
                        f"+{new_pairs} new pairs (found {len(batch_pairs)}, total {len(similar_pairs_set)} unique) | "
                        f"Speed: {rate:.1f} batches/s | ETA: {eta/60:.1f}m"
                    )
                    last_log_time = now

                # Checkpoint periodically
                if batches_since_checkpoint >= checkpoint_interval:
                    _save_checkpoint(checkpoint_file, processed_batches, similar_pairs_set)
                    batches_since_checkpoint = 0

            except TimeoutError:
                logger.error(f"Batch {batch_idx} TIMEOUT after 10 minutes")
                raise
            except Exception as e:
                logger.error(f"Batch {batch_idx} FAILED with error: {e}")
                import traceback

                logger.error(traceback.format_exc())
                raise

    # Final checkpoint
    _save_checkpoint(checkpoint_file, processed_batches, similar_pairs_set)

    elapsed = time.time() - time0
    logger.info(
        f"✓ Collected {len(similar_pairs_set)} unique duplicate links in {elapsed:.1f} seconds."
    )

    return list(similar_pairs_set)


def _process_search_batch(
    batch_idx, start_idx, end_idx, batch_ids, pb_item_ids, shard_id, total_shards, sim_threshold, k
):
    """Process a single batch of embeddings to find neighbors."""
    from models import Embedding

    # Setup connection in worker thread
    db = Embedding._meta.database
    if not db.is_closed():
        db.close()
    db.connect(reuse_if_open=True)

    # Set work_mem for this connection
    try:
        db.execute_sql("SET work_mem = '256MB'")
        db.execute_sql("SET jit = off")
    except:
        pass

    try:
        pairs = []
        t_search0 = time.time()

        # Fetch embeddings for this batch WITH vectors
        batch_embeddings = list(
            Embedding.select()
            .where(Embedding.id_embedding.in_(batch_ids))
            .order_by(Embedding.id_embedding)
        )

        if not batch_embeddings:
            logger.warning(f"Batch {batch_idx}: No embeddings found for {len(batch_ids)} IDs")
            return [], 0, 0

        # Build WHERE clause for shard filtering
        if shard_id is not None and total_shards > 1:
            # When searching, only compare within the same shard
            shard_filter = f"AND (id_embedding % {total_shards}) = {shard_id}"
        else:
            shard_filter = ""

        # Process each embedding in the batch
        for i, emb in enumerate(batch_embeddings):
            vec = emb.embedding
            if hasattr(vec, "tolist"):
                vec = vec.tolist()

            # Use raw SQL for better performance
            query_results = db.execute_sql(
                f"""
                SELECT 
                    id_embedding,
                    1 - (embedding <=> %s::vector) as similarity
                FROM embedding
                WHERE pipeline_batch_item_id = ANY(%s)
                {shard_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """,
                (vec, list(pb_item_ids), vec, k),
            ).fetchall()

            # Filter by threshold and exclude self-matches
            for neighbor_id, sim in query_results:
                if sim >= sim_threshold and neighbor_id != emb.id_embedding:
                    pair = tuple(sorted([emb.id_embedding, neighbor_id]))
                    pairs.append(pair)

        t_search1 = time.time()
        search_time = t_search1 - t_search0
        queries_per_sec = len(batch_embeddings) / search_time if search_time > 0 else 0

        return pairs, search_time, queries_per_sec

    except Exception as e:
        logger.error(f"Error in batch {batch_idx}: {e}")
        raise
    finally:
        if not db.is_closed():
            db.close()


def _save_checkpoint(checkpoint_file, processed_batches, similar_pairs_set):
    """Save progress to checkpoint file."""
    with open(checkpoint_file, "wb") as f:
        pickle.dump(
            {
                "processed_batches": processed_batches,
                "similar_pairs_set": similar_pairs_set,
            },
            f,
        )
    logger.debug(
        f"Checkpoint saved: {len(processed_batches)} batches, {len(similar_pairs_set)} unique pairs"
    )


def _cluster_embeddings(embedding_ids, similar_pairs):
    """Use Union-Find algorithm to cluster embeddings into connected components."""
    # Initialize parent pointers
    parent = {eid: eid for eid in embedding_ids}

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
    for eid in embedding_ids:
        root = find(eid)
        if root not in group_map:
            group_map[root] = []
        group_map[root].append(eid)

    # Assign group IDs (using min of group as ID for consistency)
    logger.info("Assigning group IDs...")
    dedupe_assignments = {}
    for i, (root, group_members) in enumerate(group_map.items()):
        group_id = min(group_members)  # Use min ID as group identifier
        for eid in group_members:
            dedupe_assignments[eid] = group_id

        if (i + 1) % 10000 == 0:
            logger.info(f"Assigned {i + 1:,}/{len(group_map):,} groups")

    num_groups = len(group_map)
    logger.info(f"Found {num_groups:,} unique groups from {len(embedding_ids):,} vectors.")

    return dedupe_assignments, num_groups


def _write_assignments_batched(
    embeddings_meta, dedupe_assignments, pb_item_ids, shard_id, total_shards
):
    """Write deduplication assignments to database in batches."""
    db = DedupedEmbedding._meta.database
    db.create_tables([DedupedEmbedding], safe=True)

    # When sharding, only delete assignments for this shard's embeddings
    logger.info("Clearing old assignments for this shard...")
    embedding_ids_to_delete = list(dedupe_assignments.keys())

    if embedding_ids_to_delete:
        deleted = (
            DedupedEmbedding.delete()
            .where(DedupedEmbedding.embedding_id.in_(embedding_ids_to_delete))
            .execute()
        )
        logger.info(f"Deleted {deleted} existing assignments")
    else:
        logger.info("No existing assignments to delete")

    logger.info("Writing new assignments...")
    now = datetime.utcnow()

    # Fetch embeddings in batches to get vectors
    chunk_size = 5000
    embedding_ids = list(dedupe_assignments.keys())

    total_written = 0

    with db.atomic():
        for i in range(0, len(embedding_ids), chunk_size):
            chunk_ids = embedding_ids[i : i + chunk_size]

            # Fetch embeddings with vectors
            embeddings = {
                e.id_embedding: e
                for e in Embedding.select().where(Embedding.id_embedding.in_(chunk_ids))
            }

            # Prepare rows
            output_rows = []
            for eid in chunk_ids:
                emb = embeddings.get(eid)
                if not emb:
                    logger.warning(f"Embedding {eid} not found, skipping")
                    continue

                meta = embeddings_meta[eid]

                # Convert numpy array to list for PostgreSQL
                embedding_vec = emb.embedding
                if hasattr(embedding_vec, "tolist"):
                    embedding_vec = embedding_vec.tolist()
                elif isinstance(embedding_vec, (list, tuple)):
                    # Already a list, but ensure all elements are Python floats
                    embedding_vec = [float(x) for x in embedding_vec]

                output_rows.append(
                    {
                        "embedding_id": eid,
                        "group_id": dedupe_assignments[eid],
                        "pipeline_batch_item": meta["pipeline_batch_item"],
                        "detection": meta["detection"],
                        "scan_filename": meta["scan_filename"],
                        "embedding": embedding_vec,
                        "created": now,
                    }
                )

            # Batch insert
            if output_rows:
                DedupedEmbedding.insert_many(output_rows).execute()
                total_written += len(output_rows)
                logger.info(
                    f"Wrote {total_written:,}/{len(embedding_ids):,} assignments "
                    f"({100*total_written/len(embedding_ids):.1f}%)"
                )

    logger.info(f"✓ Successfully wrote {total_written:,} deduplicated embeddings")
