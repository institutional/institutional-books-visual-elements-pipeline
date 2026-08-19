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
import time
import random
import threading
import numpy as np
import requests.exceptions
from huggingface_hub import HfApi, CommitOperationAdd
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
    HF_EXPORT_SAMPLE_LIMIT,
    HF_EXPORT_NETWORK_MAX_RETRIES,
    HF_EXPORT_NETWORK_BASE_DELAY,
    HF_EXPORT_ITEM_IDS_CACHE_PATH,
    HF_EXPORT_SHARD_SIZE,
    HF_EXPORT_ITEMS_PER_FETCH,
    HF_EXPORT_IO_WORKERS,
)

HF_REUPLOAD_REPO = "institutional/ib-ve-reupload"

_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ConnectionError,
    TimeoutError,
    OSError,
)

ORIENTABLE_CLASSES = ("Image/Illustration", "Chart/Graph")

_IMAGE_STRUCT_TYPE = pa.struct([("bytes", pa.large_binary()), ("path", pa.string())])

PARQUET_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("crop_gen", _IMAGE_STRUCT_TYPE),
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

HF_FEATURES_METADATA = json.dumps({
    "info": {
        "features": {
            "id": {"dtype": "int64", "_type": "Value"},
            "crop_gen": {"_type": "Image"},
            "barcode_src": {"dtype": "string", "_type": "Value"},
            "page_filename_src": {"dtype": "string", "_type": "Value"},
            "bbox_xyxy_gen": {"feature": {"dtype": "float64", "_type": "Value"}, "_type": "Sequence"},
            "width_gen": {"dtype": "int64", "_type": "Value"},
            "height_gen": {"dtype": "int64", "_type": "Value"},
            "pixel_count_mpx_gen": {"dtype": "float64", "_type": "Value"},
            "detection_confidence_gen": {"dtype": "float64", "_type": "Value"},
            "classification_gen": {"dtype": "string", "_type": "Value"},
            "classification_confidence_gen": {"dtype": "float64", "_type": "Value"},
            "classification_probs_gen": [{"label": {"dtype": "string", "_type": "Value"}, "prob": {"dtype": "float64", "_type": "Value"}}],
            "phash_gen": {"dtype": "string", "_type": "Value"},
            "embedding_gen": {"feature": {"dtype": "float64", "_type": "Value"}, "_type": "Sequence"},
            "caption_exp": {"dtype": "string", "_type": "Value"},
            "caption_linear_prob_exp": {"dtype": "float64", "_type": "Value"},
            "caption_lang_passed_exp": {"dtype": "string", "_type": "Value"},
            "caption_lang_detected_exp": {"dtype": "string", "_type": "Value"},
            "caption_chronam_thesauri_matches_exp": {"dtype": "string", "_type": "Value"},
            "orientation_correction_gen": {"dtype": "string", "_type": "Value"},
            "orientation_correction_confidence_gen": {"dtype": "float64", "_type": "Value"},
            "orientation_correction_probs_gen": {"dtype": "string", "_type": "Value"},
        }
    }
})


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
    if HF_EXPORT_ITEM_IDS_CACHE_PATH.exists():
        with open(HF_EXPORT_ITEM_IDS_CACHE_PATH, "r") as f:
            return json.load(f)

    import fcntl

    HF_EXPORT_ITEM_IDS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(HF_EXPORT_ITEM_IDS_CACHE_PATH) + ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if HF_EXPORT_ITEM_IDS_CACHE_PATH.exists():
                with open(HF_EXPORT_ITEM_IDS_CACHE_PATH, "r") as f:
                    return json.load(f)

            logger.info("  Fetching item IDs from DB...")
            conn = _get_raw_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT DISTINCT d.pipeline_batch_item_id
                    FROM hf_dataset_ids hf
                    JOIN detection d ON d.id_detection = hf.detection_id
                    ORDER BY d.pipeline_batch_item_id
                """)
                ids = [row[0] for row in cur.fetchall()]
                cur.close()
            finally:
                try:
                    conn.rollback()
                except Exception:
                    pass

            tmp_path = str(HF_EXPORT_ITEM_IDS_CACHE_PATH) + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(ids, f)
            os.replace(tmp_path, str(HF_EXPORT_ITEM_IDS_CACHE_PATH))
            return ids
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


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


def _build_row_group_table(records: list[dict]) -> pa.Table:
    columns = {field.name: pa.array([r[field.name] for r in records], type=field.type) for field in PARQUET_SCHEMA}
    schema_with_meta = PARQUET_SCHEMA.with_metadata({"huggingface": HF_FEATURES_METADATA})
    return pa.table(columns, schema=schema_with_meta)


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

    orientation_correction = row.get("orientation_correction_gen")
    orientation_confidence = row.get("orientation_correction_confidence_gen")
    orientation_probs_raw = row.get("orientation_correction_probs_gen")
    if orientation_probs_raw is not None:
        if isinstance(orientation_probs_raw, str):
            orientation_probs_str = orientation_probs_raw
        else:
            orientation_probs_str = json.dumps(orientation_probs_raw)
    else:
        orientation_probs_str = None

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
        "orientation_correction": orientation_correction,
        "orientation_confidence": float(orientation_confidence) if orientation_confidence is not None else None,
        "orientation_probs_str": orientation_probs_str,
    }


_thread_local = threading.local()


def _get_output_s3_client():
    if not hasattr(_thread_local, "s3"):
        _thread_local.s3 = get_s3_client("OUTPUT")
    return _thread_local.s3


def _load_and_encode_crops_for_item(
    item_id: int,
    barcode: str,
    det_ids: list[int],
    scan_filenames: dict[int, str],
    corrections: dict[int, str | None],
) -> dict[int, bytes | None]:
    """Download crops from S3, decode PNG, optionally rotate, encode WebP — all in one thread."""
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
                        det_id = expected_files[member.name]
                        png_bytes = f.read()
                        crops[det_id] = _png_to_webp(png_bytes, corrections.get(det_id))
        del tar_bytes
    except Exception as e:
        logger.warning(f"Could not load crops for item {item_id} ({barcode}): {e}")

    return crops


def _png_to_webp(png_bytes: bytes, correction: str | None = None) -> bytes | None:
    arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    if correction and correction != "upright" and correction in CV2_ROTATION_MAP:
        arr = cv2.rotate(arr, CV2_ROTATION_MAP[correction])
    success, webp_bytes = cv2.imencode(".webp", arr, [cv2.IMWRITE_WEBP_QUALITY, 95])
    if success:
        return webp_bytes.tobytes()
    return None


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
    help=f"Upload only a sample of {HF_EXPORT_SAMPLE_LIMIT} images",
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
    "--io-workers",
    type=int,
    default=HF_EXPORT_IO_WORKERS,
    help="Number of threads for S3 download + crop (default: 4)",
)
@click.option(
    "--skip-music-reclassification",
    is_flag=True,
    help="Skip music reclassification phase",
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
    io_workers,
    skip_music_reclassification,
    dry_run,
    limit,
    items_per_fetch,
):
    """
    Export filtered dataset to HuggingFace with embedded WebP crops.

    Reads from the filtered_dataset view, downloads crops from the OUTPUT S3 bucket,
    re-encodes as WebP (quality 95), applies orientation correction (rotation),
    applies music reclassification, and writes parquet shards with the crop bytes
    embedded directly in each row. Embeddings are included inline.

    Target dataset: institutional/ib-ve-reupload

    Designed for GNU parallel using --chunk-index and --total-chunks:

        seq 0 31 | parallel -j8 'uv run pipeline.py export to-hf --chunk-index {} --total-chunks 32'

    Each chunk writes its own parquet shards and uploads independently.
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

    logger.info(f"{chunk_label}Starting HuggingFace export (embedded crops)...")
    logger.info(f"  Classification confidence threshold: {classification_threshold}")
    logger.info(f"  Dataset repo: {HF_REUPLOAD_REPO}")
    if sample:
        logger.info(f"  Sample mode: {HF_EXPORT_SAMPLE_LIMIT} images")

    # --- Load music keywords ---
    music_keywords = None
    if not skip_music_reclassification:
        music_keywords = _load_music_keywords()
        logger.info(f"  Loaded music keywords for {len(music_keywords)} languages")

    # --- Fetch item IDs ---
    logger.info(f"{chunk_label}Fetching item IDs...")
    all_item_ids = _fetch_item_ids()
    total_items = len(all_item_ids)
    logger.info(f"  Total items in filtered_dataset: {total_items:,}")

    if not all_item_ids:
        logger.error("No items found. Aborting.")
        return

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
    output_path = Path(ANALYSIS_OUTPUT_DIR) / f"hf_reupload{chunk_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)

    main_output_dir = output_path / "data"
    main_output_dir.mkdir(parents=True, exist_ok=True)

    failed_crops_path = output_path / f"failed_crops{chunk_suffix}.jsonl"

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

    total_records = 0
    items_processed = 0
    music_reclassified = 0
    music_kept = 0
    skipped_crops = 0

    def _class_label_to_split_name(label: str | None) -> str:
        if not label:
            return "unknown"
        slug = label.lower()
        slug = slug.replace("/", "_").replace(" ", "_")
        return slug

    def _flush_shard(split_name: str, records: list[dict]) -> str:
        split_shard_indices[split_name] += 1
        shard_num = split_shard_indices[split_name]
        shard_filename = f"{split_name}-{shard_num:05d}-of-XXXXX.parquet"
        shard_path = main_output_dir / shard_filename
        table = _build_row_group_table(records)
        pq.write_table(table, str(shard_path))
        return str(shard_path)

    # --- Main processing loop ---
    t_start = time.time()
    done = False

    logger.info(f"  {chunk_label}I/O workers for S3 download+crop: {io_workers}")

    with ThreadPoolExecutor(max_workers=io_workers) as crop_executor:
        for fetch_start in range(0, len(my_item_ids), items_per_fetch):
            if done:
                break

            fetch_ids = my_item_ids[fetch_start:fetch_start + items_per_fetch]
            fetch_ids = [i for i in fetch_ids if i not in completed_items]
            if not fetch_ids:
                continue

            rows = _fetch_rows_for_items(fetch_ids)
            if not rows:
                continue

            grouped = _group_rows_by_item(rows)
            del rows

            item_list: list[tuple[int, str, list[dict]]] = []
            for item_id, item_rows in grouped.items():
                processed = [_extract_row_fields(row, classification_threshold) for row in item_rows]
                barcode = processed[0]["volume_barcode"] or "unknown"
                item_list.append((item_id, barcode, processed))
            del grouped

            for batch_start in range(0, len(item_list), io_workers):
                if done:
                    break

                batch = item_list[batch_start:batch_start + io_workers]
                futures = {}
                for item_id, barcode, processed in batch:
                    det_ids = [r["det_id"] for r in processed]
                    scan_fns = {r["det_id"]: r["scan_filename"] for r in processed}
                    corrections = {}
                    for r in processed:
                        correction = r["orientation_correction"]
                        if (
                            correction
                            and correction != "upright"
                            and correction in CV2_ROTATION_MAP
                            and r["classification_label"] in ORIENTABLE_CLASSES
                        ):
                            corrections[r["det_id"]] = correction
                    fut = crop_executor.submit(_load_and_encode_crops_for_item, item_id, barcode, det_ids, scan_fns, corrections)
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

                    for r in processed:
                        if done:
                            break

                        det_id = r["det_id"]
                        webp_bytes = crops.get(det_id)

                        if webp_bytes is None:
                            skipped_crops += 1
                            with open(failed_crops_path, "a") as f:
                                f.write(json.dumps({
                                    "det_id": det_id,
                                    "item_id": r["item_id"],
                                    "barcode": r["volume_barcode"],
                                }) + "\n")
                            continue

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

                        # Build parquet record with embedded crop
                        record = {
                            "id": det_id,
                            "crop_gen": {"bytes": webp_bytes, "path": f"{det_id}.webp"},
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
                            "orientation_correction_gen": r["orientation_correction"],
                            "orientation_correction_confidence_gen": r["orientation_confidence"],
                            "orientation_correction_probs_gen": r["orientation_probs_str"],
                        }

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
                f"{skipped_crops:,} failed, {rate:.0f} rec/s"
            )
            if not skip_music_reclassification:
                logger.info(f"    Music: {music_kept:,} kept, {music_reclassified:,} reclassified")

    # --- Flush remaining ---
    for split_name, remaining in split_records_buffers.items():
        if remaining:
            _flush_shard(split_name, remaining)
            logger.info(f"  {chunk_label}Wrote final shard {split_name}/{split_shard_indices[split_name]} ({len(remaining)} rows)")
    split_records_buffers.clear()

    total_shards = sum(split_shard_indices.values())
    logger.info(f"{chunk_label}Total records: {total_records} across {total_shards} shards in {len(split_shard_indices)} splits")
    logger.info(f"{chunk_label}Splits: {dict(split_shard_indices)}")
    logger.info(f"{chunk_label}Failed crops: {skipped_crops}")

    # --- Upload parquet shards ---
    if not dry_run:
        logger.info(f"{chunk_label}Uploading {total_shards} parquet shards to {HF_REUPLOAD_REPO}...")
        _retry(
            api.create_repo,
            repo_id=HF_REUPLOAD_REPO, repo_type="dataset", private=True, exist_ok=True,
            label="create HF dataset repo",
        )

        if main_output_dir.exists():
            for split_name, total_split_shards in split_shard_indices.items():
                for old_file in sorted(main_output_dir.glob(f"{split_name}-*-of-XXXXX.parquet")):
                    new_name = old_file.name.replace("XXXXX", f"{total_split_shards:05d}")
                    old_file.rename(main_output_dir / new_name)

        if main_output_dir.exists() and any(main_output_dir.glob("*.parquet")):
            shard_count = len(list(main_output_dir.glob("*.parquet")))
            total_mb = sum(f.stat().st_size for f in main_output_dir.glob("*.parquet")) / 1_000_000
            logger.info(f"  {chunk_label}Uploading folder: {shard_count} shards ({total_mb:.0f} MB)...")
            try:
                _retry(
                    api.upload_folder,
                    folder_path=str(main_output_dir),
                    path_in_repo="data",
                    repo_id=HF_REUPLOAD_REPO,
                    repo_type="dataset",
                    commit_message=f"Upload chunk{chunk_suffix} ({total_records} records, {shard_count} shards)",
                    label="upload folder to HF",
                )
                logger.success(f"  {chunk_label}Upload complete: {shard_count} shards")
            except Exception as e:
                logger.error(f"  {chunk_label}Upload failed: {e}")
        else:
            logger.warning(f"{chunk_label}No shards to upload")

    elapsed = time.time() - t_start
    logger.info(f"{chunk_label}Done in {elapsed:.0f}s")
    if skipped_crops:
        logger.warning(f"{chunk_label}Failed {skipped_crops:,} crops. See {failed_crops_path}")
    logger.success(f"{chunk_label}HuggingFace export complete!")
