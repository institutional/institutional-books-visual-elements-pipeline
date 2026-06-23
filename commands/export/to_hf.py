import click
from loguru import logger
from collections import defaultdict
import gc
import gzip
import io
import json
import multiprocessing as mp
import tarfile
from pathlib import Path
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from iso639 import Lang
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import random
import numpy as np
import requests.exceptions
from huggingface_hub import HfApi, CommitOperationAdd, batch_bucket_files
from utils import get_db
from utils.get_s3_client import get_s3_client
from commands.post_processing.orientation_correction import CV2_ROTATION_MAP
from const import (
    CLASSIFICATION_CLASS_DICT,
    ANALYSIS_OUTPUT_DIR,
    DATETIME_SLUG,
    OUTPUT_STORAGE_BUCKET_NAME,
    CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DETECTION_CONFIDENCE_THRESHOLD,
    MUSIC_CONFIDENCE_THRESHOLD,
    MODEL_CLASS_INDEX_ORDER,
    HF_EXPORT_IMAGES_REPO,
    HF_EXPORT_DATASET_REPO,
    HF_EXPORT_EMBEDDINGS_REPO,
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

ORIENTABLE_CLASSES = ("Image/Illustration", "Chart/Graph")

EMBEDDINGS_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("embedding_gen", pa.list_(pa.float64())),
])


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


def _fetch_item_ids() -> list[int]:
    """Fetch item IDs using efficient JOIN query, with local file cache."""
    if HF_EXPORT_ITEM_IDS_CACHE_PATH.exists():
        with open(HF_EXPORT_ITEM_IDS_CACHE_PATH, "r") as f:
            return json.load(f)

    conn = _get_raw_connection()
    try:
        cur = conn.cursor(name="fetch_item_ids_cursor")
        cur.itersize = 50_000
        cur.execute("""
            SELECT DISTINCT pbi.id_pipeline_batch_item
            FROM pipeline_batch_item pbi
            JOIN detection d ON d.pipeline_batch_item_id = pbi.id_pipeline_batch_item
            WHERE d.bbox_conf >= %s
            ORDER BY pbi.id_pipeline_batch_item
        """, (DETECTION_CONFIDENCE_THRESHOLD,))
        ids = []
        while True:
            batch = cur.fetchmany(50_000)
            if not batch:
                break
            ids.extend(row[0] for row in batch)
            logger.info(f"    Loaded {len(ids):,} item IDs so far...")
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


def format_classification_probs(probs) -> list[dict] | None:
    if not probs:
        return None
    if len(probs) != len(MODEL_CLASS_INDEX_ORDER):
        return None
    prob_dicts = [{"label": MODEL_CLASS_INDEX_ORDER[i], "prob": float(probs[i])} for i in range(len(probs))]
    prob_dicts.sort(key=lambda x: x["prob"], reverse=True)
    return prob_dicts


def _load_music_keywords() -> dict[str, list[str]]:
    keywords_path = Path(__file__).parent.parent.parent / "const" / "music_keywords.json"
    with open(keywords_path, "r") as f:
        raw = json.load(f)
    return {lang: [kw.lower() for kw in kws] for lang, kws in raw.items()}


def _reclassify_music_row(
    classification_confidence: float | None,
    caption_lang: str | None,
    caption_text: str | None,
    music_keywords: dict[str, list[str]],
) -> str:
    if classification_confidence is not None and classification_confidence > MUSIC_CONFIDENCE_THRESHOLD:
        return "Music"

    if not caption_lang or caption_lang not in music_keywords:
        return "Other"

    if not caption_text:
        return "Other"

    caption_lower = caption_text.lower()
    for keyword in music_keywords[caption_lang]:
        if keyword in caption_lower:
            return "Music"

    return "Other"


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
    ("orientation_correction_gen", pa.string()),
    ("orientation_correction_confidence_gen", pa.float64()),
    ("orientation_correction_probs_gen", pa.string()),
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
    bbox_xywh = row.get("bbox_xywh")
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

    caption_text = row.get("caption_text")
    caption_lang = row.get("caption_lang")
    caption_lang_detected = row.get("caption_lang_detected")
    caption_linear_prob = row.get("caption_linear_prob")
    thesaurus_matches = row.get("caption_thesaurus_matches")

    image_hash = row.get("image_hash")
    embedding = row.get("embedding")
    volume_barcode = row["barcode"]

    is_non_captionable = classification_label in ("Artifact", "Ex Libris/Decorative")
    caption_lang_passed = None
    if caption_text:
        if caption_text in ("Undetermined", "Undetermined."):
            caption_text = "CAPTION FAILED"
        else:
            caption_lang_passed = lang_name_to_iso639_3(caption_lang)
    elif is_non_captionable:
        caption_text = None
    else:
        caption_text = "CAPTION FAILED"

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

    orientation_correction = row.get("orientation_correction_gen")
    orientation_confidence = row.get("orientation_correction_confidence_gen")
    orientation_probs = row.get("orientation_correction_probs_gen")

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
        "orientation_correction_gen": orientation_correction,
        "orientation_correction_confidence_gen": float(orientation_confidence) if orientation_confidence is not None else None,
        "orientation_correction_probs_gen": json.dumps(orientation_probs) if orientation_probs is not None else None,
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


def _rotate_crop(webp_bytes: bytes, correction: str) -> bytes | None:
    """Rotate a WebP crop according to orientation correction and re-encode."""
    arr = cv2.imdecode(np.frombuffer(webp_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return webp_bytes
    rotated = cv2.rotate(arr, CV2_ROTATION_MAP[correction])
    success, out_bytes = cv2.imencode(".webp", rotated, [cv2.IMWRITE_WEBP_QUALITY, 95])
    if success:
        return out_bytes.tobytes()
    return webp_bytes


def _flush_hf_bucket_batch(
    upload_batch: list[tuple[bytes, str]],
    hf_token: str,
    max_retries: int = HF_EXPORT_NETWORK_MAX_RETRIES,
    base_timeout: int = HF_EXPORT_NETWORK_TIMEOUT,
    failed_uploads_path: Path | None = None,
) -> tuple[int, int]:
    for attempt in range(1, max_retries + 1):
        timeout = base_timeout * attempt
        result_queue = mp.Queue()

        def _worker():
            try:
                batch_bucket_files(
                    HF_EXPORT_IMAGES_REPO,
                    add=upload_batch,
                    token=hf_token,
                )
                result_queue.put(("ok", len(upload_batch), 0))
            except Exception as e:
                result_queue.put(("error", str(e), 0))

        proc = mp.Process(target=_worker)
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.kill()
            proc.join()
            if attempt < max_retries:
                delay = HF_EXPORT_NETWORK_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.warning(
                    f"  HF batch timed out after {timeout}s, "
                    f"retry {attempt}/{max_retries} in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue
            logger.error(f"  HF batch timed out after {max_retries} attempts")
            _log_failed_uploads(upload_batch, failed_uploads_path)
            return (0, len(upload_batch))

        if not result_queue.empty():
            result = result_queue.get_nowait()
            if result[0] == "ok":
                return (result[1], result[2])
            else:
                error_msg = result[1]
                if attempt < max_retries:
                    delay = HF_EXPORT_NETWORK_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
                    logger.warning(
                        f"  HF batch failed ({error_msg}), "
                        f"retry {attempt}/{max_retries} in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"  HF batch failed after {max_retries} attempts: {error_msg}")
                _log_failed_uploads(upload_batch, failed_uploads_path)
                return (0, len(upload_batch))

        if attempt < max_retries:
            delay = HF_EXPORT_NETWORK_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 2)
            logger.warning(f"  HF batch subprocess exited unexpectedly, retry {attempt}/{max_retries} in {delay:.1f}s...")
            time.sleep(delay)
            continue
        _log_failed_uploads(upload_batch, failed_uploads_path)
        return (0, len(upload_batch))

    _log_failed_uploads(upload_batch, failed_uploads_path)
    return (0, len(upload_batch))


def _log_failed_uploads(upload_batch: list[tuple[bytes, str]], failed_uploads_path: Path | None):
    if not failed_uploads_path:
        return
    with open(failed_uploads_path, "a") as f:
        for _, filename in upload_batch:
            f.write(f"{filename}\n")
    logger.error(f"  Logged {len(upload_batch)} failed uploads to {failed_uploads_path}")


@click.command("to-hf")
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
    help="Skip uploading parquet shards to HF dataset repo",
)
@click.option(
    "--skip-image-upload",
    is_flag=True,
    help="Skip uploading crop images to HF bucket",
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
    help="Check HF images bucket for existing files and skip uploading those",
)
@click.option(
    "--skip-music-reclassification",
    is_flag=True,
    help="Skip music reclassification phase",
)
@click.option(
    "--skip-embedding-separation",
    is_flag=True,
    help="Skip writing embeddings to a separate dataset",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run all processing but skip uploads to HF",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of items to process (for testing)",
)
@click.option(
    "--items-per-fetch",
    type=int,
    default=HF_EXPORT_ITEMS_PER_FETCH,
    help="Number of item IDs to fetch from the DB per batch",
)
def to_hf(
    classification_threshold,
    sample,
    shard_size,
    chunk_index,
    total_chunks,
    image_batch_size,
    skip_parquet_upload,
    skip_image_upload,
    io_workers,
    skip_existing,
    skip_music_reclassification,
    skip_embedding_separation,
    dry_run,
    limit,
    items_per_fetch,
):
    """
    Export filtered dataset to HuggingFace: crop images to a HF bucket, metadata to a
    dataset repo as parquet shards split by classification label.

    Reads from the filtered_dataset view, downloads crops from the OUTPUT S3 bucket,
    re-encodes as WebP (quality 95), applies orientation correction (rotation) from DB,
    applies music reclassification, uploads images via batch_bucket_files, writes
    parquet shards (with orientation + reclassification columns), and optionally
    separates embeddings into a dedicated dataset.

    Designed for GNU parallel using --chunk-index and --total-chunks:

        seq 0 31 | parallel -j8 'uv run pipeline.py export to-hf --chunk-index {} --total-chunks 32'

    Each chunk writes its own parquet shards and uploads its own images.
    """
    if (chunk_index is None) != (total_chunks is None):
        logger.error("--chunk-index and --total-chunks must be used together")
        return

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable is required")
        return

    api = HfApi(token=hf_token)
    chunk_label = f"[chunk {chunk_index}/{total_chunks}] " if chunk_index is not None else ""

    logger.info(f"{chunk_label}Starting HuggingFace export...")
    logger.info(f"  Classification confidence threshold: {classification_threshold}")
    if sample:
        logger.info(f"  Sample mode: {HF_EXPORT_SAMPLE_LIMIT} images")
    logger.info(f"  Images repo: {HF_EXPORT_IMAGES_REPO}")
    logger.info(f"  Dataset repo: {HF_EXPORT_DATASET_REPO}")
    if not skip_embedding_separation:
        logger.info(f"  Embeddings repo: {HF_EXPORT_EMBEDDINGS_REPO}")

    # --- Load music keywords ---
    music_keywords = None
    if not skip_music_reclassification:
        music_keywords = _load_music_keywords()
        logger.info(f"  Loaded music keywords for {len(music_keywords)} languages")

    # --- Check existing images in HF bucket ---
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

    # --- Fetch item IDs ---
    logger.info(f"{chunk_label}Fetching item IDs...")
    all_item_ids = _fetch_item_ids()
    total_items = len(all_item_ids)
    logger.info(f"  Total items in filtered_dataset: {total_items:,}")

    if not all_item_ids:
        logger.error("No items found. Aborting.")
        return

    # Partition items for GNU parallel
    if chunk_index is not None:
        items_per_chunk = (total_items + total_chunks - 1) // total_chunks
        start = chunk_index * items_per_chunk
        end = min(start + items_per_chunk, total_items)
        my_item_ids = all_item_ids[start:end]
        logger.info(f"  {chunk_label}This chunk: items {start:,}-{end:,} ({len(my_item_ids):,} items)")
    else:
        my_item_ids = all_item_ids

    if sample:
        my_item_ids = my_item_ids[:HF_EXPORT_SAMPLE_LIMIT]

    if limit:
        my_item_ids = my_item_ids[:limit]
        logger.info(f"  Limited to {limit} items")

    del all_item_ids

    if not my_item_ids:
        logger.info(f"{chunk_label}No items to process.")
        return

    # --- Output setup ---
    chunk_suffix = f"_chunk{chunk_index:04d}" if chunk_index is not None else ""
    output_path = Path(ANALYSIS_OUTPUT_DIR) / f"hf_export_{DATETIME_SLUG}{chunk_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)

    main_output_dir = output_path / "data"
    main_output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_output_dir = output_path / "embeddings_data"
    if not skip_embedding_separation:
        embeddings_output_dir.mkdir(parents=True, exist_ok=True)

    failed_uploads_path = output_path / "failed_uploads.txt"

    # Checkpoint for resumability
    checkpoint_path = output_path / "checkpoint.json"
    completed_items: set[int] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            completed_items = set(json.load(f).get("completed_items", []))
        logger.info(f"  Resuming: {len(completed_items)} items already completed")

    # Buffers
    split_records_buffers: dict[str, list[dict]] = defaultdict(list)
    split_shard_indices: dict[str, int] = defaultdict(int)
    embeddings_buffer: list[dict] = []
    embeddings_shard_idx = 0

    total_records = 0
    upload_count = 0
    upload_failures = 0
    items_processed = 0
    music_reclassified = 0
    music_kept = 0
    image_upload_batch: list[tuple[bytes, str]] = []

    def _flush_shard(split_name: str, records: list[dict]) -> str:
        split_shard_indices[split_name] += 1
        shard_num = split_shard_indices[split_name]
        shard_filename = f"{split_name}-{shard_num:05d}-of-XXXXX.parquet"
        shard_path = main_output_dir / shard_filename
        table = _build_row_group_table(records)
        pq.write_table(table, str(shard_path))
        return str(shard_path)

    def _flush_embeddings_shard(records: list[dict]):
        nonlocal embeddings_shard_idx
        embeddings_shard_idx += 1
        shard_filename = f"embeddings{chunk_suffix}-{embeddings_shard_idx:05d}.parquet"
        shard_path = embeddings_output_dir / shard_filename
        columns = {
            "id": pa.array([r["id"] for r in records], type=pa.int64()),
            "embedding_gen": pa.array([r["embedding_gen"] for r in records], type=pa.list_(pa.float64())),
        }
        table = pa.table(columns, schema=EMBEDDINGS_SCHEMA)
        pq.write_table(table, str(shard_path))

    def _flush_image_batch():
        nonlocal upload_count, upload_failures
        if not image_upload_batch:
            return
        if skip_image_upload or dry_run:
            image_upload_batch.clear()
            return
        add_pairs = [(img_bytes, filename) for filename, img_bytes in image_upload_batch]
        batch_mb = sum(len(b) for b, _ in add_pairs) / 1_000_000
        logger.info(f"  {chunk_label}Uploading batch of {len(add_pairs)} images ({batch_mb:.0f} MB)...")
        image_upload_batch.clear()
        success, failures = _flush_hf_bucket_batch(
            add_pairs, hf_token,
            failed_uploads_path=failed_uploads_path,
        )
        upload_count += success
        upload_failures += failures
        if success:
            logger.info(f"  {chunk_label}Uploaded {success} images ({upload_count} total)")
        if failures:
            logger.error(f"  {chunk_label}Failed {failures} images")
        del add_pairs
        gc.collect()

    # --- Main processing loop ---
    t_start = time.time()
    done = False

    logger.info(f"  {chunk_label}I/O workers for S3 download+crop: {io_workers}")

    with ThreadPoolExecutor(max_workers=io_workers) as crop_executor:
        for fetch_start in range(0, len(my_item_ids), items_per_fetch):
            if done:
                break

            fetch_ids = my_item_ids[fetch_start:fetch_start + items_per_fetch]

            # Skip already-completed items (checkpoint)
            fetch_ids = [i for i in fetch_ids if i not in completed_items]
            if not fetch_ids:
                continue

            rows = _fetch_rows_for_items(fetch_ids)
            if not rows:
                continue

            grouped = _group_rows_by_item(rows)
            del rows

            # Pre-process rows
            item_list: list[tuple[int, str, list[dict]]] = []
            for item_id, item_rows in grouped.items():
                processed = [_extract_row_fields(row, classification_threshold) for row in item_rows]
                barcode = processed[0]["volume_barcode"] or "unknown"
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

                    # Determine which images to skip uploading
                    skip_upload_det_ids: set[int] = set()
                    if existing_images:
                        for r in processed:
                            fn = _make_crop_filename(barcode, r["scan_filename"], r["det_id"])
                            if fn in existing_images:
                                skip_upload_det_ids.add(r["det_id"])

                    for r in processed:
                        if done:
                            break

                        det_id = r["det_id"]
                        crop_bytes = crops.get(det_id)

                        # Apply orientation rotation if needed
                        if crop_bytes is not None:
                            correction = r["orientation_correction_gen"]
                            if (
                                correction
                                and correction != "upright"
                                and correction in CV2_ROTATION_MAP
                                and r["classification_label"] in ORIENTABLE_CLASSES
                            ):
                                rotated = _rotate_crop(crop_bytes, correction)
                                if rotated is not None:
                                    crop_bytes = rotated

                        # Queue image for upload (if we have bytes and not skipping)
                        if crop_bytes is not None and det_id not in skip_upload_det_ids:
                            crop_filename = _make_crop_filename(barcode, r["scan_filename"], det_id)
                            image_upload_batch.append((crop_filename, crop_bytes))
                            if len(image_upload_batch) >= image_batch_size:
                                _flush_image_batch()

                        # Apply music reclassification
                        classification_label = r["classification_label"]
                        if not skip_music_reclassification and classification_label == "Music" and music_keywords:
                            new_label = _reclassify_music_row(
                                classification_confidence=r["classification_confidence"],
                                caption_lang=r["caption_lang_passed"],
                                caption_text=r["caption_text"],
                                music_keywords=music_keywords,
                            )
                            if new_label != "Music":
                                classification_label = "Other"
                                music_reclassified += 1
                            else:
                                music_kept += 1

                        # Build parquet record
                        crop_filename = _make_crop_filename(barcode, r["scan_filename"], det_id)
                        crop_url = _make_crop_hf_url(crop_filename)

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
                            "classification_gen": classification_label,
                            "classification_confidence_gen": r["classification_confidence"],
                            "classification_probs_gen": r["classification_probs"],
                            "phash_gen": r["phash"],
                            "embedding_gen": r["embedding"],
                            "caption_exp": r["caption_text"],
                            "caption_linear_prob_exp": r["caption_linear_prob"],
                            "caption_lang_passed_exp": r["caption_lang_passed"],
                            "caption_lang_detected_exp": r["caption_lang_detected"],
                            "caption_chronam_thesauri_matches_exp": r["thesaurus_str"],
                            "orientation_correction_gen": r["orientation_correction_gen"],
                            "orientation_correction_confidence_gen": r["orientation_correction_confidence_gen"],
                            "orientation_correction_probs_gen": r["orientation_correction_probs_gen"],
                        }

                        # Embedding separation
                        if not skip_embedding_separation and r["embedding"] is not None:
                            embeddings_buffer.append({"id": det_id, "embedding_gen": r["embedding"]})
                            if len(embeddings_buffer) >= shard_size:
                                _flush_embeddings_shard(embeddings_buffer[:shard_size])
                                del embeddings_buffer[:shard_size]

                        # Buffer by classification split
                        split_name = _class_label_to_split_name(classification_label)
                        split_records_buffers[split_name].append(record)
                        total_records += 1

                        if len(split_records_buffers[split_name]) >= shard_size:
                            _flush_shard(split_name, split_records_buffers[split_name])
                            logger.info(f"  {chunk_label}Wrote shard {split_name}/{split_shard_indices[split_name]} ({shard_size} rows)")
                            split_records_buffers[split_name] = []

                        if sample and total_records >= HF_EXPORT_SAMPLE_LIMIT:
                            done = True

                    completed_items.add(item_id)
                    del crops
                del futures

            del item_list
            gc.collect()

            # Write checkpoint
            tmp_checkpoint = str(checkpoint_path) + ".tmp"
            with open(tmp_checkpoint, "w") as f:
                json.dump({"completed_items": list(completed_items)}, f)
            os.replace(tmp_checkpoint, str(checkpoint_path))

            elapsed = time.time() - t_start
            rate = total_records / elapsed if elapsed > 0 else 0
            logger.info(
                f"  {chunk_label}Progress: {items_processed:,} items, {total_records:,} records, "
                f"{upload_count:,} uploaded, {rate:.0f} rec/s"
            )
            if not skip_music_reclassification:
                logger.info(f"    Music: {music_kept:,} kept, {music_reclassified:,} reclassified")

    # --- Flush remaining ---
    _flush_image_batch()

    for split_name, remaining in split_records_buffers.items():
        if remaining:
            _flush_shard(split_name, remaining)
            logger.info(f"  {chunk_label}Wrote final shard {split_name}/{split_shard_indices[split_name]} ({len(remaining)} rows)")
    split_records_buffers.clear()

    if embeddings_buffer:
        _flush_embeddings_shard(embeddings_buffer)
    embeddings_buffer.clear()

    total_shards = sum(split_shard_indices.values())
    logger.info(f"{chunk_label}Total records: {total_records} across {total_shards} shards in {len(split_shard_indices)} splits")
    logger.info(f"{chunk_label}Splits: {dict(split_shard_indices)}")
    logger.info(f"{chunk_label}Images uploaded: {upload_count}, failures: {upload_failures}")

    # --- Upload parquet shards ---
    if not skip_parquet_upload and not dry_run:
        logger.info(f"{chunk_label}Uploading {total_shards} parquet shards to {HF_EXPORT_DATASET_REPO}...")
        _retry(
            api.create_repo,
            repo_id=HF_EXPORT_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True,
            label="create HF dataset repo",
        )

        if main_output_dir.exists():
            for split_name, total_split_shards in split_shard_indices.items():
                for old_file in sorted(main_output_dir.glob(f"{split_name}-*-of-XXXXX.parquet")):
                    new_name = old_file.name.replace("XXXXX", f"{total_split_shards:05d}")
                    old_file.rename(main_output_dir / new_name)

        operations = []
        if main_output_dir.exists():
            for pf in sorted(main_output_dir.glob("*.parquet")):
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

    # --- Upload embeddings ---
    if not skip_embedding_separation and not dry_run:
        logger.info(f"{chunk_label}Uploading embeddings to {HF_EXPORT_EMBEDDINGS_REPO}...")
        _retry(
            api.create_repo,
            repo_id=HF_EXPORT_EMBEDDINGS_REPO, repo_type="dataset", private=True, exist_ok=True,
            label="create embeddings dataset repo",
        )

        emb_operations = []
        if embeddings_output_dir.exists():
            for pf in sorted(embeddings_output_dir.glob("*.parquet")):
                with open(pf, "rb") as f:
                    file_bytes = f.read()
                emb_operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"data/{pf.name}",
                        path_or_fileobj=file_bytes,
                    )
                )

        if emb_operations:
            try:
                _retry(
                    api.create_commit,
                    repo_id=HF_EXPORT_EMBEDDINGS_REPO,
                    repo_type="dataset",
                    operations=emb_operations,
                    commit_message=f"Upload embeddings{chunk_suffix} ({embeddings_shard_idx} shards)",
                    label="upload embeddings dataset",
                )
                logger.success(f"{chunk_label}Embeddings uploaded to {HF_EXPORT_EMBEDDINGS_REPO}")
            except Exception as e:
                logger.error(f"{chunk_label}Failed to upload embeddings after retries: {e}")

    elapsed = time.time() - t_start
    logger.info(f"{chunk_label}Done in {elapsed:.0f}s")
    logger.success(f"{chunk_label}HuggingFace export complete!")
