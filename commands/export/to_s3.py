import click
from loguru import logger
from collections import defaultdict
import gc
import gzip
import io
import json
import tarfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from iso639 import Lang
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import threading
import time
from utils import get_db
from utils.get_s3_client import get_s3_client
from const import (
    CLASSIFICATION_CLASS_DICT,
    ANALYSIS_OUTPUT_DIR,
    DATETIME_SLUG,
    OUTPUT_STORAGE_BUCKET_NAME,
    FILTER_STORAGE_BUCKET_NAME,
    DETECTION_CONFIDENCE_THRESHOLD,
    CLASSIFICATION_CONFIDENCE_THRESHOLD,
    MODEL_CLASS_INDEX_ORDER,
    S3_EXPORT_ROW_GROUP_SIZE,
    S3_EXPORT_MULTIPART_THRESHOLD,
    S3_EXPORT_MULTIPART_CHUNK_SIZE,
    S3_EXPORT_MULTIPART_PARALLEL_PARTS,
    S3_EXPORT_SHARD_SIZE,
    S3_EXPORT_MAX_INFLIGHT,
    HF_EXPORT_ITEM_IDS_CACHE_PATH,
)


def _get_raw_connection():
    import psycopg2
    db = get_db()
    conn = db.connection()
    try:
        conn.cursor().execute("SELECT 1")
        conn.rollback()
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        db.close()
        db.connect()
        conn = db.connection()
    if conn.autocommit:
        conn.autocommit = False
    return conn


def _fetch_item_ids_paginated() -> list[int]:
    if HF_EXPORT_ITEM_IDS_CACHE_PATH.exists():
        with open(HF_EXPORT_ITEM_IDS_CACHE_PATH, "r") as f:
            return json.load(f)

    conn = _get_raw_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT pipeline_batch_item_id
            FROM filtered_dataset
            ORDER BY pipeline_batch_item_id
        """)
        ids = [row[0] for row in cur.fetchall()]
        cur.close()
    finally:
        try:
            conn.rollback()
        except Exception:
            pass

    HF_EXPORT_ITEM_IDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(HF_EXPORT_ITEM_IDS_CACHE_PATH) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(ids, f)
    os.replace(tmp_path, str(HF_EXPORT_ITEM_IDS_CACHE_PATH))
    return ids


def _fetch_rows_for_items(item_ids: list[int]) -> list[dict]:
    if not item_ids:
        return []
    conn = _get_raw_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT *
            FROM filtered_dataset
            WHERE pipeline_batch_item_id = ANY(%s)
            ORDER BY pipeline_batch_item_id
        """, (item_ids,))
        rows = cur.fetchall()
        if not rows:
            cur.close()
            return []
        col_names = [desc[0] for desc in cur.description]
        cur.close()
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        try:
            conn.rollback()
        except Exception:
            pass


def _group_rows_by_item(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        item_id = row["pipeline_batch_item_id"]
        if item_id not in grouped:
            grouped[item_id] = []
        grouped[item_id].append(row)
    return grouped


def lang_name_to_iso639_3(name: str) -> str | None:
    if not name:
        return None
    try:
        return Lang(name).pt3
    except Exception:
        return None


def format_classification_probs(probs) -> list[dict] | None:
    if not probs:
        return None
    if len(probs) != len(MODEL_CLASS_INDEX_ORDER):
        return None
    prob_dicts = [{"label": MODEL_CLASS_INDEX_ORDER[i], "prob": float(probs[i])} for i in range(len(probs))]
    prob_dicts.sort(key=lambda x: x["prob"], reverse=True)
    return prob_dicts


PARQUET_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("crop_gen", pa.large_binary()),
    ("barcode_src", pa.string()),
    ("page_filename_src", pa.string()),
    ("bbox_xyxy_gen", pa.list_(pa.float64())),
    ("width_gen", pa.int64()),
    ("height_gen", pa.int64()),
    ("pixel_count_mpx_gen", pa.float64()),
    ("detection_confidence_gen", pa.float64()),
    ("classification_gen", pa.string()),
    ("classification_confidence_gen", pa.float64()),
    ("classification_probs_gen", pa.list_(pa.struct([("label", pa.string()), ("prob", pa.float64())]))),
    ("phash_gen", pa.string()),
    ("embedding_gen", pa.list_(pa.float64())),
    ("caption_exp", pa.string()),
    ("caption_linear_prob_exp", pa.float64()),
    ("caption_lang_passed_exp", pa.string()),
    ("caption_lang_detected_exp", pa.string()),
    ("caption_chronam_thesauri_matches_exp", pa.map_(pa.string(), pa.map_(pa.string(), pa.int64()))),
])


def _build_row_group_table(records: list[dict]) -> pa.Table:
    columns = {field.name: pa.array([r[field.name] for r in records], type=field.type) for field in PARQUET_SCHEMA}
    return pa.table(columns, schema=PARQUET_SCHEMA)


def _extract_row_fields(row: dict, classification_threshold: float) -> dict:
    det_id = row["id_detection"]
    item_id = row["pipeline_batch_item_id"]
    scan_fn = row["scan_filename"]
    bbox_xyxy = row["bbox_xyxy"]
    bbox_xywh = row["bbox_xywh"]
    bbox_conf = row["bbox_conf"]

    if bbox_xyxy and not isinstance(bbox_xyxy, list):
        bbox_xyxy = list(bbox_xyxy)
    if bbox_xywh and not isinstance(bbox_xywh, list):
        bbox_xywh = list(bbox_xywh)

    pred_class = row["pred_class"]
    pred_conf = row["classification_conf"]
    probs = row["classification_probs"]

    if pred_conf is not None and pred_conf < classification_threshold:
        pred_class = "Other"
    classification_label = CLASSIFICATION_CLASS_DICT.get(pred_class, pred_class) if pred_class else None

    caption_text = row["caption_text"]
    caption_lang = row["caption_lang"]
    caption_lang_detected = row["caption_lang_detected"]
    caption_linear_prob = row["caption_linear_prob"]
    thesaurus_matches = row["caption_thesaurus_matches"]

    image_hash = row["image_hash"]
    embedding = row["embedding"]
    volume_barcode = row["barcode"]

    is_non_captionable = classification_label in ("Artifact", "Ex Libris/Decorative")
    if caption_text:
        if caption_text in ("Undetermined", "Undetermined."):
            caption_text = "CAPTION FAILED"
        caption_lang_passed = lang_name_to_iso639_3(caption_lang)
    elif is_non_captionable:
        caption_text = None
        caption_lang_passed = None
    else:
        caption_text = "CAPTION FAILED"
        caption_lang_passed = None

    caption_is_valid = caption_text is not None and caption_text != "CAPTION FAILED"
    if not caption_is_valid:
        caption_linear_prob = None
        caption_lang_detected = None
        thesaurus_matches = None
        caption_lang_passed = None

    if isinstance(thesaurus_matches, str) and thesaurus_matches == "null":
        thesaurus_matches = None

    if bbox_xywh and len(bbox_xywh) >= 4:
        width = int(round(bbox_xywh[2]))
        height = int(round(bbox_xywh[3]))
        pixel_count_mpx = (width * height) / 1_000_000
    else:
        width = None
        height = None
        pixel_count_mpx = None

    classification_probs_formatted = format_classification_probs(probs)

    embedding_list = None
    if embedding is not None:
        if isinstance(embedding, str):
            embedding_list = [float(x) for x in embedding.strip("[]").split(",")]
        else:
            embedding_list = [float(x) for x in embedding]

    return {
        "det_id": det_id,
        "item_id": item_id,
        "scan_filename": scan_fn,
        "bbox_xyxy": bbox_xyxy,
        "bbox_conf": float(bbox_conf) if bbox_conf is not None else None,
        "volume_barcode": str(volume_barcode) if volume_barcode else None,
        "width": width,
        "height": height,
        "pixel_count_mpx": pixel_count_mpx,
        "classification_label": classification_label,
        "classification_confidence": float(pred_conf) if pred_conf is not None else None,
        "classification_probs": classification_probs_formatted,
        "phash": image_hash,
        "embedding": embedding_list,
        "caption_text": caption_text,
        "caption_linear_prob": float(caption_linear_prob) if caption_linear_prob is not None else None,
        "caption_lang_passed": caption_lang_passed,
        "caption_lang_detected": caption_lang_detected,
        "thesaurus_matches": thesaurus_matches,
    }


_thread_local = threading.local()


def _get_output_s3_client():
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = get_s3_client("OUTPUT")
    return _thread_local.s3


def _load_crops_for_item(item_id: int, barcode: str, det_ids: list[int], scan_filenames: dict[int, str]) -> dict[int, bytes | None]:
    s3 = _get_output_s3_client()
    s3_key = f"crops/{item_id}/{barcode}.tar.gz"

    expected_files: dict[str, int] = {}
    for det_id in det_ids:
        scan_base = scan_filenames[det_id].rsplit(".", 1)[0]
        expected_files[f"{scan_base}_{det_id}.png"] = det_id

    crops: dict[int, bytes | None] = {d: None for d in det_ids}
    try:
        response = s3.get_object(Bucket=OUTPUT_STORAGE_BUCKET_NAME, Key=s3_key)
        gz_bytes = response["Body"].read()
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes), mode="rb") as gz:
            tar_bytes = gz.read()
        del gz_bytes
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
            for member in tar.getmembers():
                if member.name in expected_files:
                    f = tar.extractfile(member)
                    if f:
                        crops[expected_files[member.name]] = f.read()
        del tar_bytes
    except Exception as e:
        logger.warning(f"Could not load crops for item {item_id} ({barcode}): {e}")

    return crops


def _upload_parquet_to_s3(s3_client, parquet_path: str, s3_key: str, bucket_name: str) -> bool:
    try:
        file_size = os.path.getsize(parquet_path)

        if file_size < S3_EXPORT_MULTIPART_THRESHOLD:
            with open(parquet_path, "rb") as fh:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=fh,
                    ContentType="application/octet-stream",
                )
        else:
            total_parts = (file_size + S3_EXPORT_MULTIPART_CHUNK_SIZE - 1) // S3_EXPORT_MULTIPART_CHUNK_SIZE
            logger.info(f"  Multipart upload {s3_key} ({file_size / 1024 / 1024:.0f} MB, {total_parts} parts)")
            mpu = s3_client.create_multipart_upload(
                Bucket=bucket_name,
                Key=s3_key,
                ContentType="application/octet-stream",
            )
            upload_id = mpu["UploadId"]
            try:
                part_specs = []
                part_number = 1
                offset = 0
                while offset < file_size:
                    length = min(S3_EXPORT_MULTIPART_CHUNK_SIZE, file_size - offset)
                    part_specs.append((part_number, offset, length))
                    offset += length
                    part_number += 1

                parts = [None] * len(part_specs)

                def _upload_one_part(spec):
                    pn, off, ln = spec
                    with open(parquet_path, "rb") as fh:
                        fh.seek(off)
                        chunk = fh.read(ln)
                    resp = s3_client.upload_part(
                        Bucket=bucket_name,
                        Key=s3_key,
                        PartNumber=pn,
                        UploadId=upload_id,
                        Body=chunk,
                    )
                    return pn, resp["ETag"]

                with ThreadPoolExecutor(max_workers=S3_EXPORT_MULTIPART_PARALLEL_PARTS) as part_executor:
                    for pn, etag in part_executor.map(_upload_one_part, part_specs):
                        parts[pn - 1] = {"ETag": etag, "PartNumber": pn}

                s3_client.complete_multipart_upload(
                    Bucket=bucket_name,
                    Key=s3_key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
            except Exception:
                s3_client.abort_multipart_upload(
                    Bucket=bucket_name, Key=s3_key, UploadId=upload_id
                )
                raise
        return True
    except Exception as e:
        logger.error(f"Error uploading {s3_key}: {e}")
        return False
    finally:
        try:
            os.unlink(parquet_path)
        except OSError:
            pass


ITEMS_PER_FETCH = 50


@click.command("to-s3")
@click.option("--shard-size", type=int, default=S3_EXPORT_SHARD_SIZE, help="Rows per parquet shard")
@click.option("--classification-threshold", type=float, default=CLASSIFICATION_CONFIDENCE_THRESHOLD)
@click.option("--chunk-index", type=int, default=None, help="Which chunk to process (0-indexed). Use with --total-chunks for GNU parallel.")
@click.option("--total-chunks", type=int, default=None, help="Total number of chunks. Use with --chunk-index for GNU parallel.")
@click.option("--io-workers", type=int, default=S3_EXPORT_MAX_INFLIGHT, help="Threads for S3 crop download")
@click.option("--prefix", type=str, default=None, help="S3 key prefix for shard files (default: generated datetime slug)")
@click.option("--sample", type=int, default=None, help="Limit to N items for testing")
def to_s3(shard_size, classification_threshold, chunk_index, total_chunks, io_workers, prefix, sample):
    """
    Export filtered dataset to S3 as parquet shards with embedded PNG crops.

    Use GNU parallel for parallelism:

        seq 0 31 | parallel -j8 'python main.py export to-s3 --chunk-index {} --total-chunks 32'
    """
    if (chunk_index is None) != (total_chunks is None):
        logger.error("--chunk-index and --total-chunks must be used together")
        return

    chunk_label = f"[chunk {chunk_index}/{total_chunks}] " if chunk_index is not None else ""

    if prefix is None:
        prefix = DATETIME_SLUG

    logger.info(f"{chunk_label}Starting S3 export...")
    logger.info(f"  Classification threshold: {classification_threshold}")
    logger.info(f"  Shard size: {shard_size}")
    logger.info(f"  S3 prefix: {prefix}")
    logger.info(f"  I/O workers: {io_workers}")

    logger.info(f"{chunk_label}Fetching item IDs...")
    all_item_ids = _fetch_item_ids_paginated()
    total_items = len(all_item_ids)
    logger.info(f"  Total items in filtered_dataset: {total_items:,}")

    if chunk_index is not None:
        items_per_chunk = (total_items + total_chunks - 1) // total_chunks
        start = chunk_index * items_per_chunk
        end = min(start + items_per_chunk, total_items)
        my_item_ids = all_item_ids[start:end]
        logger.info(f"  This chunk: items {start:,}-{end:,} ({len(my_item_ids):,} items)")
    else:
        my_item_ids = all_item_ids

    if sample:
        my_item_ids = my_item_ids[:sample]
        logger.info(f"  Sample mode: {len(my_item_ids)} items")

    del all_item_ids

    s3_client = get_s3_client("FILTER")
    bucket_name = FILTER_STORAGE_BUCKET_NAME

    # Shard state
    shard_prefix = f"{prefix}-c{chunk_index:02d}" if chunk_index is not None else prefix

    # Resume: find existing shards for this chunk's prefix to determine how many records to skip
    existing_shards = 0
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=shard_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    existing_shards += 1
    except Exception as e:
        logger.warning(f"  Could not list existing shards: {e}")

    records_to_skip = existing_shards * shard_size
    shard_idx = existing_shards
    shard_start_idx = existing_shards

    if existing_shards:
        logger.info(f"  {chunk_label}Resuming: found {existing_shards} existing shards, skipping {records_to_skip:,} records")

    row_group_buf: list[dict] = []
    row_group_crop_bytes = 0
    shard_record_count = 0
    current_writer: pq.ParquetWriter | None = None
    current_tmp_path: str | None = None
    current_s3_key: str | None = None

    total_records = 0
    skipped_records = 0
    items_processed = 0
    upload_executor = ThreadPoolExecutor(max_workers=4)
    upload_futures = []

    def _open_shard():
        nonlocal shard_idx, shard_record_count, current_writer, current_tmp_path, current_s3_key
        shard_idx += 1
        current_s3_key = f"{shard_prefix}-{shard_idx:04d}.parquet"
        tmp_fd, current_tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(tmp_fd)
        current_writer = pq.ParquetWriter(current_tmp_path, PARQUET_SCHEMA)
        shard_record_count = 0

    def _flush_row_group():
        nonlocal row_group_buf, shard_record_count, row_group_crop_bytes
        if not row_group_buf:
            return
        current_writer.write_table(_build_row_group_table(row_group_buf))
        shard_record_count += len(row_group_buf)
        row_group_buf = []
        row_group_crop_bytes = 0

    def _close_shard():
        nonlocal current_writer, current_tmp_path, current_s3_key
        if current_writer is None:
            return
        _flush_row_group()
        current_writer.close()
        current_writer = None

        tmp_path = current_tmp_path
        s3_key = current_s3_key
        current_tmp_path = None
        current_s3_key = None

        tmp_size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        logger.info(f"  {chunk_label}Closed shard {shard_idx} ({shard_record_count} rows, {tmp_size_mb:.0f} MB) -> uploading")

        # Drain completed futures
        still_pending = []
        for f, key in upload_futures:
            if f.done():
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"  Upload failed for {key}: {e}")
            else:
                still_pending.append((f, key))
        upload_futures[:] = still_pending

        # Backpressure
        while len(upload_futures) >= 8:
            for f, _ in upload_futures:
                if not f.done():
                    f.result(timeout=60)
                    break
            upload_futures[:] = [(f, k) for f, k in upload_futures if not f.done()]

        fut = upload_executor.submit(_upload_parquet_to_s3, s3_client, tmp_path, s3_key, bucket_name)
        upload_futures.append((fut, s3_key))

    t_start = time.time()

    # Fast-skip items covered by existing shards without fetching from DB or S3
    items_to_skip = 0
    if records_to_skip > 0:
        logger.info(f"  {chunk_label}Counting items to skip...")
        cumulative = 0
        for fetch_start in range(0, len(my_item_ids), ITEMS_PER_FETCH):
            if cumulative >= records_to_skip:
                break
            fetch_ids = my_item_ids[fetch_start:fetch_start + ITEMS_PER_FETCH]
            conn = _get_raw_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM filtered_dataset WHERE pipeline_batch_item_id = ANY(%s)", (fetch_ids,))
                batch_count = cur.fetchone()[0]
                cur.close()
            finally:
                conn.rollback()
            if cumulative + batch_count <= records_to_skip:
                cumulative += batch_count
                items_to_skip += len(fetch_ids)
            else:
                break
        skipped_records = cumulative
        logger.info(f"  {chunk_label}Skipping {items_to_skip:,} items ({skipped_records:,} records)")

    active_item_ids = my_item_ids[items_to_skip:]

    with ThreadPoolExecutor(max_workers=io_workers) as crop_executor:
        for fetch_start in range(0, len(active_item_ids), ITEMS_PER_FETCH):
            fetch_ids = active_item_ids[fetch_start:fetch_start + ITEMS_PER_FETCH]
            rows = _fetch_rows_for_items(fetch_ids)
            if not rows:
                continue

            grouped = _group_rows_by_item(rows)
            del rows

            item_work: list[tuple[int, str, list[dict]]] = []
            for item_id, item_rows in grouped.items():
                processed = [_extract_row_fields(row, classification_threshold) for row in item_rows]
                barcode = processed[0]["volume_barcode"] or "unknown"
                item_work.append((item_id, barcode, processed))
            del grouped

            for batch_start in range(0, len(item_work), io_workers):
                batch = item_work[batch_start:batch_start + io_workers]
                futures = {}
                for item_id, barcode, processed in batch:
                    det_ids = [r["det_id"] for r in processed]
                    scan_fns = {r["det_id"]: r["scan_filename"] for r in processed}
                    fut = crop_executor.submit(_load_crops_for_item, item_id, barcode, det_ids, scan_fns)
                    futures[fut] = (item_id, barcode, processed)

                for fut in as_completed(futures):
                    item_id, barcode, processed = futures[fut]
                    try:
                        crops = fut.result()
                    except Exception as e:
                        logger.warning(f"  {chunk_label}Crop failed for item {item_id}: {e}")
                        crops = {}

                    items_processed += 1

                    for r in processed:
                        det_id = r["det_id"]
                        crop_bytes = crops.get(det_id)
                        if crop_bytes is None:
                            continue

                        # Skip remaining records from the partial-skip boundary
                        if skipped_records < records_to_skip:
                            skipped_records += 1
                            continue

                        embedding = r["embedding"]

                        record = {
                            "id": det_id,
                            "crop_gen": crop_bytes,
                            "barcode_src": r["volume_barcode"],
                            "page_filename_src": r["scan_filename"],
                            "bbox_xyxy_gen": r["bbox_xyxy"],
                            "width_gen": r["width"],
                            "height_gen": r["height"],
                            "pixel_count_mpx_gen": r["pixel_count_mpx"],
                            "detection_confidence_gen": r["bbox_conf"],
                            "classification_gen": r["classification_label"],
                            "classification_confidence_gen": r["classification_confidence"],
                            "classification_probs_gen": r["classification_probs"],
                            "phash_gen": r["phash"],
                            "embedding_gen": embedding,
                            "caption_exp": r["caption_text"],
                            "caption_linear_prob_exp": r["caption_linear_prob"],
                            "caption_lang_passed_exp": r["caption_lang_passed"],
                            "caption_lang_detected_exp": r["caption_lang_detected"],
                            "caption_chronam_thesauri_matches_exp": r["thesaurus_matches"],
                        }

                        if current_writer is None:
                            _open_shard()

                        row_group_buf.append(record)
                        row_group_crop_bytes += len(crop_bytes)
                        total_records += 1

                        if len(row_group_buf) >= S3_EXPORT_ROW_GROUP_SIZE or row_group_crop_bytes >= 1_900_000_000:
                            _flush_row_group()

                        if shard_record_count + len(row_group_buf) >= shard_size:
                            _close_shard()

                    del crops
                del futures

            del item_work
            gc.collect()

            elapsed = time.time() - t_start
            rate = total_records / elapsed if elapsed > 0 else 0
            logger.info(f"  {chunk_label}Progress: {items_processed:,} items, {total_records:,} records, {rate:.0f} rec/s")

    # Close final shard
    if current_writer is not None:
        _close_shard()

    # Wait for uploads
    logger.info(f"{chunk_label}Waiting for remaining uploads...")
    for f, key in upload_futures:
        try:
            f.result()
        except Exception as e:
            logger.error(f"  Upload failed for {key}: {e}")
    upload_futures.clear()
    upload_executor.shutdown(wait=False)

    total_shards = shard_idx - shard_start_idx
    elapsed = time.time() - t_start
    logger.success(f"{chunk_label}Done: {total_records:,} records across {total_shards} shards in {elapsed:.0f}s")
