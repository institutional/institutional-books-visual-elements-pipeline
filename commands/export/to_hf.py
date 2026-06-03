import click
from loguru import logger
from collections import defaultdict
import gc
import gzip
import io
import json
import tarfile
from pathlib import Path
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from iso639 import Lang
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import time
import random
import numpy as np
import requests.exceptions
from huggingface_hub import HfApi, CommitOperationAdd, batch_bucket_files, sync_bucket
from utils import decode_image_bytes, get_db
from utils.get_s3_client import get_s3_client
from utils.get_cache import get_cache
from const import (
    CLASSIFICATION_CLASS_DICT,
    ANALYSIS_OUTPUT_DIR,
    DATETIME_SLUG,
    OUTPUT_STORAGE_BUCKET_NAME,
    DETECTION_CONFIDENCE_THRESHOLD,
    CLASSIFICATION_CONFIDENCE_THRESHOLD,
    MODEL_CLASS_INDEX_ORDER,
    HF_EXPORT_IMAGES_REPO,
    HF_EXPORT_DATASET_REPO,
    HF_EXPORT_SAMPLE_LIMIT,
    HF_EXPORT_NETWORK_TIMEOUT,
    HF_EXPORT_NETWORK_MAX_RETRIES,
    HF_EXPORT_NETWORK_BASE_DELAY,
    HF_EXPORT_ITEM_IDS_CACHE_PATH,
    HF_EXPORT_SHARD_SIZE,
    HF_EXPORT_IMAGE_BATCH_SIZE,
    HF_EXPORT_ITEMS_PER_FETCH,
    HF_EXPORT_IO_WORKERS,
)

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION", "64")

_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ConnectionError,
    TimeoutError,
    OSError,
)

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    cause = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
    if cause and isinstance(cause, _RETRYABLE_EXCEPTIONS):
        return True
    return False


def _retry(fn, *args, max_retries=HF_EXPORT_NETWORK_MAX_RETRIES, label="network call", **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_retryable(e) and attempt < max_retries:
                delay = HF_EXPORT_NETWORK_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.warning(f"{label} failed ({type(e).__name__}), retry {attempt}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise


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


def _fetch_chunk(resume_after_item_id: int | None, limit: int) -> tuple[list[dict], list[str] | None]:
    conn = _get_raw_connection()
    try:
        cur = conn.cursor()
        if resume_after_item_id is not None:
            cur.execute("""
                SELECT *
                FROM filtered_dataset
                WHERE pipeline_batch_item_id > %s
                ORDER BY pipeline_batch_item_id
                LIMIT %s
            """, (resume_after_item_id, limit))
        else:
            cur.execute("""
                SELECT *
                FROM filtered_dataset
                ORDER BY pipeline_batch_item_id
                LIMIT %s
            """, (limit,))

        rows = cur.fetchall()
        if not rows:
            cur.close()
            return [], None
        col_names = [desc[0] for desc in cur.description]
        cur.close()
        return [dict(zip(col_names, row)) for row in rows], col_names
    finally:
        try:
            conn.rollback()
        except Exception:
            pass

def _fetch_item_ids_paginated() -> list[int]:
    """Fetch all distinct pipeline_batch_item_ids in order, with local file cache."""
    if HF_EXPORT_ITEM_IDS_CACHE_PATH.exists():
        with open(HF_EXPORT_ITEM_IDS_CACHE_PATH, "r") as f:
            ids = json.load(f)
        return ids

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
    """Fetch all rows for a specific set of item IDs."""
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


def generate_crop_webp_bytes(bbox_xyxy: list[float], scan_image) -> bytes | None:
    try:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        h, w = scan_image.shape[:2]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        crop_array = scan_image[y1:y2, x1:x2, :]
        success, webp_bytes = cv2.imencode(".webp", crop_array, [cv2.IMWRITE_WEBP_QUALITY, 95])
        if success:
            return webp_bytes.tobytes()
    except Exception:
        pass
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
    ("crop_gen", pa.string()),
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
    ("caption_chronam_thesauri_matches_exp", pa.string()),
])

def _build_row_group_table(records: list[dict]) -> pa.Table:
    columns = {field.name: pa.array([r[field.name] for r in records], type=field.type) for field in PARQUET_SCHEMA}
    return pa.table(columns, schema=PARQUET_SCHEMA)


def _class_label_to_split_name(label: str | None) -> str:
    if not label:
        return "unknown"
    slug = label.lower()
    slug = slug.replace("/", "_").replace(" ", "_")
    return slug


def _make_crop_filename(barcode: str, page_filename: str, detection_id: int) -> str:
    page = Path(page_filename).stem if page_filename else "unknown"
    return f"{barcode}_{page}_{detection_id}.webp"


def _make_crop_hf_url(filename: str) -> str:
    return f"https://huggingface.co/buckets/{HF_EXPORT_IMAGES_REPO}/resolve/{filename}"


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
    thesaurus_str = json.dumps(thesaurus_matches) if thesaurus_matches else None

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
        "thesaurus_str": thesaurus_str,
    }


def _load_and_crop_item(item_id: int, barcode: str, rows: list[dict]) -> dict[int, bytes | None]:
    """Load pre-existing crops from OUTPUT bucket and re-encode as WebP."""
    s3 = get_s3_client("OUTPUT")
    s3_key = f"crops/{item_id}/{barcode}.tar.gz"

    def _s3_download():
        resp = s3.get_object(Bucket=OUTPUT_STORAGE_BUCKET_NAME, Key=s3_key)
        return resp["Body"].read()

    try:
        gz_bytes = _retry(_s3_download, label=f"OUTPUT download {barcode}")
    except Exception:
        return {}

    png_by_det_id: dict[int, bytes] = {}
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes), mode="rb") as gz:
            tar_bytes = gz.read()
        del gz_bytes
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".png"):
                    continue
                stem = Path(member.name).stem
                parts = stem.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    det_id = int(parts[1])
                    with tar.extractfile(member) as fh:
                        png_by_det_id[det_id] = fh.read()
        del tar_bytes
    except Exception:
        return {}

    crops: dict[int, bytes | None] = {}
    for r in rows:
        det_id = r["det_id"]
        png_bytes = png_by_det_id.get(det_id)
        if png_bytes is None:
            crops[det_id] = None
            continue
        try:
            arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                crops[det_id] = None
                continue
            success, webp_bytes = cv2.imencode(".webp", arr, [cv2.IMWRITE_WEBP_QUALITY, 95])
            crops[det_id] = webp_bytes.tobytes() if success else None
        except Exception:
            crops[det_id] = None

    del png_by_det_id
    return crops


@click.command("to-hf")
@click.option(
    "--detection-threshold",
    type=float,
    default=DETECTION_CONFIDENCE_THRESHOLD,
    help=f"Minimum detection confidence threshold (default: {DETECTION_CONFIDENCE_THRESHOLD})",
)
@click.option(
    "--classification-threshold",
    type=float,
    default=CLASSIFICATION_CONFIDENCE_THRESHOLD,
    help=f"Classification confidence below this becomes 'other' (default: {CLASSIFICATION_CONFIDENCE_THRESHOLD})",
)
@click.option(
    "--sample",
    is_flag=True,
    help=f"Upload only a sample of {HF_EXPORT_SAMPLE_LIMIT} images to start with",
)
@click.option(
    "--shard-size",
    type=int,
    default=HF_EXPORT_SHARD_SIZE,
    help="Number of rows per parquet shard (default: 5000)",
)
@click.option(
    "--chunk-index",
    type=int,
    default=None,
    help="Which chunk to process (0-indexed). Use with --total-chunks for GNU parallel.",
)
@click.option(
    "--total-chunks",
    type=int,
    default=None,
    help="Total number of chunks to split work into. Use with --chunk-index for GNU parallel.",
)
@click.option(
    "--image-batch-size",
    type=int,
    default=HF_EXPORT_IMAGE_BATCH_SIZE,
    help="Number of images per upload batch (default: 200)",
)
@click.option(
    "--skip-parquet-upload",
    is_flag=True,
    help="Skip uploading parquet shards to HF dataset repo (useful when combining shards later)",
)
@click.option(
    "--io-workers",
    type=int,
    default=HF_EXPORT_IO_WORKERS,
    help="Number of threads for S3 download + crop (default: 4)",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Check HF images bucket for existing files and skip downloading/processing those items",
)
def to_hf(
    detection_threshold,
    classification_threshold,
    sample,
    shard_size,
    chunk_index,
    total_chunks,
    image_batch_size,
    skip_parquet_upload,
    io_workers,
    skip_existing,
):
    """
    Export filtered dataset to HuggingFace.

    Processes items sequentially — use GNU parallel for parallelism:

        seq 0 31 | parallel -j8 'python main.py export to-hf --chunk-index {} --total-chunks 32'

    Each chunk writes its own parquet shards and uploads its own images.
    Combine shards afterward with a final upload step (--skip-images on a single run).

    Images: institutional/institutional-books-hl-visual-elements-images
    Dataset: institutional/institutional-books-hl-visual-elements
    """
    if (chunk_index is None) != (total_chunks is None):
        logger.error("--chunk-index and --total-chunks must be used together")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable is required")
        return

    api = HfApi(token=hf_token)

    existing_images: set[str] = set()
    if skip_existing:
        existing_cache_path = Path(ANALYSIS_OUTPUT_DIR) / "hf_existing_images.txt"
        if existing_cache_path.exists():
            logger.info(f"Loading existing image list from cache: {existing_cache_path}")
            with open(existing_cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_images.add(line)
            logger.info(f"  Loaded {len(existing_images):,} existing images from cache")
        else:
            logger.info("Fetching existing image list from HF bucket (one-time)...")
            try:
                for entry in api.list_bucket_tree(HF_EXPORT_IMAGES_REPO, recursive=True):
                    if hasattr(entry, "path"):
                        existing_images.add(entry.path)
                existing_cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = str(existing_cache_path) + ".tmp"
                with open(tmp_path, "w") as f:
                    for name in sorted(existing_images):
                        f.write(name + "\n")
                os.replace(tmp_path, str(existing_cache_path))
                logger.info(f"  Cached {len(existing_images):,} existing images to {existing_cache_path}")
            except Exception as e:
                logger.warning(f"  Failed to list existing images: {e}. Proceeding without skip.")
                existing_images.clear()

    record_limit = HF_EXPORT_SAMPLE_LIMIT if sample else None
    chunk_label = f"[chunk {chunk_index}/{total_chunks}] " if chunk_index is not None else ""

    logger.info(f"{chunk_label}Starting HuggingFace export...")
    logger.info(f"  Detection confidence threshold: {detection_threshold}")
    logger.info(f"  Classification confidence threshold: {classification_threshold}")
    if sample:
        logger.info(f"  Sample mode: {HF_EXPORT_SAMPLE_LIMIT} images")
    logger.info(f"  Images repo: {HF_EXPORT_IMAGES_REPO}")
    logger.info(f"  Dataset repo: {HF_EXPORT_DATASET_REPO}")

    # Determine which items this chunk handles
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
        my_item_ids = my_item_ids[:HF_EXPORT_SAMPLE_LIMIT]

    del all_item_ids

    # Output setup — each chunk gets its own output dir
    chunk_suffix = f"_chunk{chunk_index:04d}" if chunk_index is not None else ""
    output_path = Path(ANALYSIS_OUTPUT_DIR) / f"hf_export_{DATETIME_SLUG}{chunk_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)

    split_records_buffers: dict[str, list[dict]] = defaultdict(list)
    split_shard_indices: dict[str, int] = defaultdict(int)

    total_records = 0
    upload_count = 0
    upload_failures = 0
    items_processed = 0
    image_upload_batch: list[tuple[str, bytes]] = []
    failed_filenames: list[str] = []

    def _flush_shard(split_name: str, records: list[dict]) -> str:
        split_shard_indices[split_name] += 1
        shard_num = split_shard_indices[split_name]
        data_dir = output_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        shard_filename = f"{split_name}-{shard_num:05d}-of-XXXXX.parquet"
        shard_path = data_dir / shard_filename
        table = _build_row_group_table(records)
        pq.write_table(table, str(shard_path))
        return str(shard_path)

    def _upload_in_subprocess(add_pairs: list[tuple[bytes, str]]) -> tuple[int, int]:
        """Run batch_bucket_files in a child process to avoid xet memory leaks."""
        import multiprocessing
        result_queue = multiprocessing.Queue()

        def _worker():
            try:
                _retry(
                    batch_bucket_files,
                    HF_EXPORT_IMAGES_REPO,
                    add=add_pairs,
                    token=hf_token,
                    label=f"upload batch of {len(add_pairs)} images",
                )
                result_queue.put((len(add_pairs), 0))
            except Exception as e:
                logger.error(f"  {chunk_label}Upload subprocess failed: {e}")
                result_queue.put((0, len(add_pairs)))

        proc = multiprocessing.Process(target=_worker)
        proc.start()
        proc.join(timeout=HF_EXPORT_NETWORK_TIMEOUT)
        if proc.is_alive():
            proc.kill()
            proc.join()
            return (0, len(add_pairs))
        if not result_queue.empty():
            return result_queue.get_nowait()
        return (0, len(add_pairs))

    def _flush_image_batch():
        nonlocal upload_count, upload_failures
        if not image_upload_batch:
            return
        batch_filenames = [filename for filename, _ in image_upload_batch]
        add_pairs = [(img_bytes, filename) for filename, img_bytes in image_upload_batch]
        image_upload_batch.clear()
        batch_mb = sum(len(b) for b, _ in add_pairs) / 1_000_000
        logger.info(f"  {chunk_label}Uploading batch of {len(add_pairs)} images ({batch_mb:.0f} MB)...")
        success, failures = _upload_in_subprocess(add_pairs)
        upload_count += success
        upload_failures += failures
        if success:
            logger.info(f"  {chunk_label}Uploaded {success} images ({upload_count} total)")
        if failures:
            logger.error(f"  {chunk_label}Failed {failures} images")
            failed_filenames.extend(batch_filenames)
        del add_pairs
        gc.collect()

    # Process items — S3 download+crop is threaded but memory-bounded
    t_start = time.time()
    done = False

    logger.info(f"  {chunk_label}I/O workers for S3 download+crop: {io_workers}")

    with ThreadPoolExecutor(max_workers=io_workers) as crop_executor:
        for fetch_start in range(0, len(my_item_ids), HF_EXPORT_ITEMS_PER_FETCH):
            if done:
                break

            fetch_ids = my_item_ids[fetch_start:fetch_start + HF_EXPORT_ITEMS_PER_FETCH]
            rows = _fetch_rows_for_items(fetch_ids)
            if not rows:
                continue

            grouped = _group_rows_by_item(rows)
            del rows

            # Pre-process rows but submit only io_workers items at a time
            item_list: list[tuple[int, str, list[dict]]] = []
            for item_id, item_rows in grouped.items():
                processed = [_extract_row_fields(row, classification_threshold) for row in item_rows]
                barcode = processed[0]["volume_barcode"] or "unknown"
                # Skip entire item if all its images already exist in the bucket
                if existing_images:
                    all_exist = all(
                        _make_crop_filename(barcode, r["scan_filename"], r["det_id"]) in existing_images
                        for r in processed
                    )
                    if all_exist:
                        items_processed += 1
                        continue
                item_list.append((item_id, barcode, processed))
            del grouped

            # Process in sliding window of io_workers to bound memory
            for batch_start in range(0, len(item_list), io_workers):
                if done:
                    break

                batch = item_list[batch_start:batch_start + io_workers]
                futures = {}
                for item_id, barcode, processed in batch:
                    fut = crop_executor.submit(_load_and_crop_item, item_id, barcode, processed)
                    futures[fut] = (item_id, barcode, processed)

                for fut in as_completed(futures):
                    if done:
                        break

                    item_id, barcode, processed = futures[fut]
                    try:
                        crops = fut.result()
                    except Exception as e:
                        logger.warning(f"  {chunk_label}Crop failed for item {item_id}: {e}")
                        crops = {}

                    items_processed += 1

                    # When --skip-existing, determine which det_ids to skip
                    skip_det_ids: set[int] = set()
                    if existing_images:
                        for r in processed:
                            fn = _make_crop_filename(barcode, r["scan_filename"], r["det_id"])
                            if fn in existing_images:
                                skip_det_ids.add(r["det_id"])

                    for r in processed:
                        if done:
                            break

                        det_id = r["det_id"]

                        if det_id in skip_det_ids:
                            continue

                        crop_bytes = crops.get(det_id)
                        if crop_bytes is None:
                            continue

                        crop_filename = _make_crop_filename(barcode, r["scan_filename"], det_id)
                        crop_url = _make_crop_hf_url(crop_filename)

                        image_upload_batch.append((crop_filename, crop_bytes))
                        if len(image_upload_batch) >= image_batch_size:
                            _flush_image_batch()

                        record = {
                            "id": det_id,
                            "crop_gen": crop_url,
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
                            "embedding_gen": r["embedding"],
                            "caption_exp": r["caption_text"],
                            "caption_linear_prob_exp": r["caption_linear_prob"],
                            "caption_lang_passed_exp": r["caption_lang_passed"],
                            "caption_lang_detected_exp": r["caption_lang_detected"],
                            "caption_chronam_thesauri_matches_exp": r["thesaurus_str"],
                        }

                        split_name = _class_label_to_split_name(r["classification_label"])
                        split_records_buffers[split_name].append(record)
                        total_records += 1

                        if len(split_records_buffers[split_name]) >= shard_size:
                            _flush_shard(split_name, split_records_buffers[split_name])
                            logger.info(f"  {chunk_label}Wrote shard {split_name}/{split_shard_indices[split_name]} ({shard_size} rows)")
                            split_records_buffers[split_name] = []

                        if record_limit and total_records >= record_limit:
                            done = True

                    del crops
                del futures

            del item_list
            gc.collect()

            elapsed = time.time() - t_start
            rate = total_records / elapsed if elapsed > 0 else 0
            logger.info(f"  {chunk_label}Progress: {items_processed:,} items, {total_records:,} records, "
                        f"{upload_count:,} uploaded, {rate:.0f} rec/s")

    # Flush remaining images
    _flush_image_batch()

    # Flush remaining records per split
    for split_name, remaining in split_records_buffers.items():
        if remaining:
            _flush_shard(split_name, remaining)
            logger.info(f"  {chunk_label}Wrote final shard {split_name}/{split_shard_indices[split_name]} ({len(remaining)} rows)")
    split_records_buffers.clear()

    total_shards = sum(split_shard_indices.values())
    logger.info(f"{chunk_label}Total records: {total_records} across {total_shards} shards in {len(split_shard_indices)} splits")
    logger.info(f"{chunk_label}Splits: {dict(split_shard_indices)}")
    logger.info(f"{chunk_label}Images uploaded: {upload_count}, failures: {upload_failures}")

    if failed_filenames:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        failed_log_path = logs_dir / f"failed_crops{chunk_suffix}.log"
        with open(failed_log_path, "w") as f:
            for fn in failed_filenames:
                f.write(fn + "\n")
        logger.warning(f"{chunk_label}Wrote {len(failed_filenames)} failed filenames to {failed_log_path}")

    # Upload dataset parquet shards to HF dataset repo
    if not skip_parquet_upload:
        logger.info(f"{chunk_label}Uploading {total_shards} parquet shards to {HF_EXPORT_DATASET_REPO}...")
        _retry(
            api.create_repo,
            repo_id=HF_EXPORT_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True,
            label="create HF dataset repo",
        )

        data_dir = output_path / "data"
        if data_dir.exists():
            for split_name, total_split_shards in split_shard_indices.items():
                for old_file in sorted(data_dir.glob(f"{split_name}-*-of-XXXXX.parquet")):
                    new_name = old_file.name.replace("XXXXX", f"{total_split_shards:05d}")
                    old_file.rename(data_dir / new_name)

        operations = []
        if data_dir.exists():
            for pf in sorted(data_dir.glob("*.parquet")):
                with open(pf, "rb") as f:
                    file_bytes = f.read()
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"data/{pf.name}",
                        path_or_fileobj=file_bytes,
                    )
                )

        if operations:
            try:
                _retry(
                    api.create_commit,
                    repo_id=HF_EXPORT_DATASET_REPO,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Upload dataset chunk{chunk_suffix} ({total_records} records, {total_shards} shards)",
                    label="dataset commit upload",
                )
                logger.success(f"{chunk_label}Dataset uploaded to {HF_EXPORT_DATASET_REPO}")
            except Exception as e:
                logger.error(f"{chunk_label}Failed to upload dataset after retries: {e}")

    elapsed = time.time() - t_start
    logger.info(f"{chunk_label}Done in {elapsed:.0f}s")
    logger.success(f"{chunk_label}HuggingFace export complete!")