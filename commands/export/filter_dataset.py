import click
from loguru import logger
from collections import defaultdict
import gc
import json
import orjson
from pathlib import Path
import openai
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from iso639 import Lang
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import io
import tempfile
import os

from models import (
    Detection,
    Classification,
    Caption,
    DedupedHash,
    DedupedEmbedding,
    PipelineBatchItem,
    ImageHash,
    ImageEmbedding,
)
from utils import decode_image_bytes, get_db
from utils.get_s3_client import get_s3_client
from const import (
    CLASSIFICATION_CLASS_DICT,
    ANALYSIS_OUTPUT_DIR,
    DATETIME_SLUG,
    OPENAI_REQUEST_TIMEOUT,
    CPUS_LIMIT,
    FILTER_STORAGE_BUCKET_NAME,
)

# Thresholds
DETECTION_CONFIDENCE_THRESHOLD = 0.75
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.70

SERVER_SIDE_CURSOR_SIZE = 100_000


def _get_raw_connection():
    """Get the underlying psycopg2 connection from peewee, with autocommit off for named cursors."""
    db = get_db()
    conn = db.connection()
    if conn.autocommit:
        conn.autocommit = False
    else:
        conn.rollback()
    return conn


def _raw_fetch_detections(detection_threshold: float, limit: int | None = None) -> dict[int, dict]:
    """Fetch detections using a server-side cursor. Returns {det_id: {fields...}}."""
    import time

    conn = _get_raw_connection()
    sql = "SELECT id_detection, pipeline_batch_item_id, scan_filename, bbox_xyxy, bbox_xywh, bbox_conf FROM detection WHERE bbox_conf >= %s"
    if limit:
        sql += f" LIMIT {limit}"

    detections_data = {}
    with conn.cursor(name="det_cursor") as cursor:
        cursor.itersize = SERVER_SIDE_CURSOR_SIZE
        logger.info("  Executing detection query...")
        t0 = time.time()
        cursor.execute(sql, (detection_threshold,))
        logger.info(f"  Query started (took {time.time() - t0:.1f}s to begin streaming)")
        count = 0
        t_last = time.time()
        while True:
            rows = cursor.fetchmany(SERVER_SIDE_CURSOR_SIZE)
            if not rows:
                break
            for row in rows:
                det_id, item_id, scan_fn, bbox_xyxy, bbox_xywh, bbox_conf = row
                detections_data[det_id] = {
                    "id_detection": det_id,
                    "pipeline_batch_item_id": item_id,
                    "scan_filename": scan_fn,
                    "bbox_xyxy": list(bbox_xyxy) if bbox_xyxy else None,
                    "bbox_xywh": list(bbox_xywh) if bbox_xywh else None,
                    "bbox_conf": float(bbox_conf) if bbox_conf is not None else None,
                }
            count += len(rows)
            now = time.time()
            if now - t_last >= 10:
                elapsed = now - t0
                rate = count / elapsed if elapsed > 0 else 0
                logger.info(f"    ... {count:,} detections loaded ({rate:,.0f} rows/sec, {elapsed:.0f}s elapsed)")
                t_last = now
    elapsed = time.time() - t0
    logger.info(f"  Detections loaded: {count:,} in {elapsed:.1f}s")
    return detections_data


def _raw_fetch_dedupe_groups(table_name: str) -> dict[int, int]:
    """Fetch detection_id -> group_id mapping from a dedupe table using server-side cursor."""
    conn = _get_raw_connection()
    sql = f"SELECT detection_id, group_id FROM {table_name}"
    mapping = {}
    with conn.cursor(name=f"{table_name}_cursor") as cursor:
        cursor.itersize = SERVER_SIDE_CURSOR_SIZE
        cursor.execute(sql)
        count = 0
        while True:
            rows = cursor.fetchmany(SERVER_SIDE_CURSOR_SIZE)
            if not rows:
                break
            for det_id, group_id in rows:
                mapping[det_id] = group_id
            count += len(rows)
            if count % 5_000_000 == 0:
                logger.info(f"    ... loaded {count:,} rows from {table_name}")
    return mapping


def _raw_fetch_dedupe_intersection() -> dict[tuple[int, int], list[int]]:
    """
    Compute dedupe intersection groups via a SQL JOIN instead of loading both tables into memory.
    Returns {(hash_group, emb_group): [detection_ids]}.
    """
    import time

    conn = _get_raw_connection()
    sql = """
        SELECT dh.detection_id, dh.group_id, de.group_id
        FROM deduped_hash dh
        INNER JOIN deduped_embedding de ON dh.detection_id = de.detection_id
    """
    intersection_groups = defaultdict(list)
    with conn.cursor(name="dedupe_join_cursor") as cursor:
        cursor.itersize = SERVER_SIDE_CURSOR_SIZE
        logger.info("  Executing JOIN query...")
        t0 = time.time()
        cursor.execute(sql)
        logger.info(f"  Query started (took {time.time() - t0:.1f}s to begin streaming)")
        count = 0
        t_last = time.time()
        while True:
            rows = cursor.fetchmany(SERVER_SIDE_CURSOR_SIZE)
            if not rows:
                break
            for det_id, hash_group, emb_group in rows:
                intersection_groups[(hash_group, emb_group)].append(det_id)
            count += len(rows)
            now = time.time()
            if now - t_last >= 10:
                elapsed = now - t0
                rate = count / elapsed if elapsed > 0 else 0
                logger.info(f"    ... {count:,} rows processed ({rate:,.0f} rows/sec, {len(intersection_groups):,} groups, {elapsed:.0f}s elapsed)")
                t_last = now
    elapsed = time.time() - t0
    logger.info(f"  JOIN complete: {count:,} rows -> {len(intersection_groups):,} groups in {elapsed:.1f}s")
    return intersection_groups


def get_dedupe_intersection_groups(detection_ids: list[int] | None = None):
    """
    Find the intersection of hash and embedding dedupe groups.
    When detection_ids is provided, only load groups for those detections.
    Returns a dict mapping (hash_group, emb_group) -> [detection_ids]
    """
    if detection_ids is not None:
        det_id_set = set(detection_ids)
        logger.info(f"Loading dedupe groups for {len(det_id_set)} detections...")

        hash_by_detection = {}
        for i in range(0, len(detection_ids), BATCH_SIZE):
            chunk = detection_ids[i : i + BATCH_SIZE]
            for dh in DedupedHash.select(DedupedHash.detection, DedupedHash.group_id).where(
                DedupedHash.detection.in_(chunk)
            ):
                hash_by_detection[dh.detection_id] = dh.group_id

        logger.info(f"  Found {len(hash_by_detection)} detections with hash groups")

        emb_by_detection = {}
        for i in range(0, len(detection_ids), BATCH_SIZE):
            chunk = detection_ids[i : i + BATCH_SIZE]
            for de in DedupedEmbedding.select(DedupedEmbedding.detection, DedupedEmbedding.group_id).where(
                DedupedEmbedding.detection.in_(chunk)
            ):
                emb_by_detection[de.detection_id] = de.group_id

        logger.info(f"  Found {len(emb_by_detection)} detections with embedding groups")
    else:
        logger.info("Loading all hash dedupe groups...")
        hash_by_detection = {}
        for dh in DedupedHash.select(DedupedHash.detection, DedupedHash.group_id):
            hash_by_detection[dh.detection_id] = dh.group_id

        logger.info(f"  Found {len(hash_by_detection)} detections with hash groups")

        logger.info("Loading all embedding dedupe groups...")
        emb_by_detection = {}
        for de in DedupedEmbedding.select(DedupedEmbedding.detection, DedupedEmbedding.group_id):
            emb_by_detection[de.detection_id] = de.group_id

        logger.info(f"  Found {len(emb_by_detection)} detections with embedding groups")

    # Find intersection: detection must be in both hash and embedding groups
    common_detections = set(hash_by_detection.keys()) & set(emb_by_detection.keys())
    logger.info(f"  {len(common_detections)} detections have both hash and embedding groups")

    # Create intersection groups: (hash_group, emb_group) tuple -> unique id
    intersection_groups = defaultdict(list)
    for det_id in common_detections:
        key = (hash_by_detection[det_id], emb_by_detection[det_id])
        intersection_groups[key].append(det_id)

    logger.info(f"  Found {len(intersection_groups)} unique intersection groups")

    return intersection_groups


def select_representative(detection_ids: list, detections_data: dict) -> int:
    """
    Select a representative detection from a group.
    Strategy: pick the one with highest detection confidence.
    """
    best_det_id = None
    best_conf = -1

    for det_id in sorted(detection_ids):
        if det_id in detections_data:
            conf = detections_data[det_id].get("bbox_conf", 0) or 0
            if conf > best_conf:
                best_conf = conf
                best_det_id = det_id

    return best_det_id if best_det_id else detection_ids[0]


def run_moderation_fn(client: openai.OpenAI, text: str) -> dict | None:
    """
    Run a single text through OpenAI's moderation API.
    Returns the moderation result or None on error.
    """
    if not text or not text.strip():
        return None
    try:
        response = client.moderations.create(input=text)
        return response.model_dump()
    except Exception as e:
        logger.warning(f"Moderation API error: {e}")
        return None


def lang_name_to_iso639_3(name: str) -> str | None:
    """Convert a human-readable language name (e.g. "English") to ISO 639-3 code (e.g. "eng")."""
    if not name:
        return None
    try:
        return Lang(name).pt3
    except Exception:
        return None


def generate_crop_png_bytes(bbox_xyxy: list[float], scan_image) -> bytes | None:
    """Generate uncompressed PNG bytes for a crop defined by bbox_xyxy."""
    try:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        h, w = scan_image.shape[:2]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        crop_array = scan_image[y1:y2, x1:x2, :]
        success, png_bytes = cv2.imencode(".png", crop_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if success:
            return png_bytes.tobytes()
    except Exception:
        pass
    return None


MODEL_CLASS_INDEX_ORDER = [
    "Artifact",
    "Chart/Graph",
    "Ex Libris/Decorative",
    "Image/Illustration",
    "Music",
]


def format_classification_probs(probs: list[float] | None) -> list[dict] | None:
    """
    Convert classification probabilities list to sorted list of {"label": str, "prob": float} dicts.
    Sorted by confidence descending.

    Uses MODEL_CLASS_INDEX_ORDER which matches the YOLO model's actual class index order.
    """
    if not probs:
        return None

    if len(probs) != len(MODEL_CLASS_INDEX_ORDER):
        return None

    prob_dicts = [{"label": MODEL_CLASS_INDEX_ORDER[i], "prob": probs[i]} for i in range(len(probs))]
    prob_dicts.sort(key=lambda x: x["prob"], reverse=True)

    return prob_dicts


def generate_record_id(det_id: int) -> int:
    """
    Generate a unique identifier for a record.
    Returns the detection id.
    """
    return det_id


PARQUET_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("crop_gen", pa.large_binary()),
    ("barcode_src", pa.string()),
    ("page_filename_src", pa.string()),
    ("bbox_xyxy_gen", pa.list_(pa.float64())),
    ("width_gen", pa.float64()),
    ("height_gen", pa.float64()),
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

ROW_GROUP_SIZE = 500


def _build_row_group_table(records: list[dict]) -> pa.Table:
    """Build a PyArrow table for a small row group chunk."""
    columns = {field.name: pa.array([r[field.name] for r in records], type=field.type) for field in PARQUET_SCHEMA}
    return pa.table(columns, schema=PARQUET_SCHEMA)


MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100 MB
MULTIPART_CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB
MULTIPART_PARALLEL_PARTS = 6


def _upload_parquet_file_to_r2(s3_client, parquet_path: str, s3_key: str, bucket_name: str) -> bool:
    """Upload a parquet file from disk to R2, streaming chunks to avoid holding in memory. Deletes the temp file when done."""
    try:
        file_size = os.path.getsize(parquet_path)

        if file_size < MULTIPART_THRESHOLD:
            with open(parquet_path, "rb") as fh:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=fh,
                    ContentType="application/octet-stream",
                )
        else:
            total_parts = (file_size + MULTIPART_CHUNK_SIZE - 1) // MULTIPART_CHUNK_SIZE
            logger.info(f"  Multipart upload {s3_key} ({file_size / 1024 / 1024:.0f} MB, {total_parts} parts, {MULTIPART_PARALLEL_PARTS} parallel)")
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
                    length = min(MULTIPART_CHUNK_SIZE, file_size - offset)
                    part_specs.append((part_number, offset, length))
                    offset += length
                    part_number += 1

                parts = [None] * len(part_specs)

                def _upload_one_part(spec):
                    pn, off, length = spec
                    with open(parquet_path, "rb") as fh:
                        fh.seek(off)
                        chunk = fh.read(length)
                    resp = s3_client.upload_part(
                        Bucket=bucket_name,
                        Key=s3_key,
                        PartNumber=pn,
                        UploadId=upload_id,
                        Body=chunk,
                    )
                    return pn, resp["ETag"]

                with ThreadPoolExecutor(max_workers=MULTIPART_PARALLEL_PARTS) as part_executor:
                    for pn, etag in part_executor.map(_upload_one_part, part_specs):
                        parts[pn - 1] = {"ETag": etag, "PartNumber": pn}

                logger.info(f"    All {len(parts)} parts uploaded for {s3_key}")
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


BATCH_SIZE = 500_000


def _batched_fetch(model_class, id_field, ids: list[int]) -> list:
    """Fetch rows in batches to avoid exceeding PostgreSQL's memory limits on large IN clauses."""
    rows = []
    for i in range(0, len(ids), BATCH_SIZE):
        chunk = ids[i : i + BATCH_SIZE]
        rows.extend(list(model_class.select().where(id_field.in_(chunk))))
    return rows


def _batched_iter(model_class, id_field, ids: list[int]):
    """Yield rows in batches without accumulating all into a list."""
    for i in range(0, len(ids), BATCH_SIZE):
        chunk = ids[i : i + BATCH_SIZE]
        yield from model_class.select().where(id_field.in_(chunk))


def _fetch_db_data_parallel(detection_ids: list[int], detections_data: dict, workers: int):
    """Fetch classifications, captions, hashes, embeddings, and volume info in parallel."""

    results = {}

    def fetch_classifications():
        return _batched_fetch(Classification, Classification.detection, detection_ids)

    def fetch_captions():
        return _batched_fetch(Caption, Caption.detection, detection_ids)

    def fetch_image_hashes():
        return _batched_fetch(ImageHash, ImageHash.detection, detection_ids)

    def fetch_image_embeddings():
        return _batched_fetch(ImageEmbedding, ImageEmbedding.detection, detection_ids)

    def fetch_volume_barcodes():
        item_ids = list(set(d["pipeline_batch_item_id"] for d in detections_data.values()))
        rows = []
        for i in range(0, len(item_ids), BATCH_SIZE):
            chunk = item_ids[i : i + BATCH_SIZE]
            rows.extend(list(
                PipelineBatchItem.select(
                    PipelineBatchItem.id_pipeline_batch_item,
                    PipelineBatchItem.ib_volume,
                ).where(PipelineBatchItem.id_pipeline_batch_item.in_(chunk))
            ))
        return rows

    logger.info(f"Fetching DB data in parallel with {workers} workers ({len(detection_ids)} detection IDs in batches of {BATCH_SIZE})...")

    with ThreadPoolExecutor(max_workers=min(workers, 5)) as executor:
        future_cls = executor.submit(fetch_classifications)
        future_cap = executor.submit(fetch_captions)
        future_hash = executor.submit(fetch_image_hashes)
        future_emb = executor.submit(fetch_image_embeddings)
        future_vol = executor.submit(fetch_volume_barcodes)

        classifications = future_cls.result()
        captions = future_cap.result()
        image_hashes = future_hash.result()
        image_embeddings = future_emb.result()
        volume_items = future_vol.result()

    logger.info(f"  Found {len(classifications)} classifications")
    logger.info(f"  Found {len(captions)} captions")
    logger.info(f"  Found {len(image_hashes)} image hashes")
    logger.info(f"  Found {len(image_embeddings)} image embeddings")

    results["classifications"] = classifications
    results["captions"] = captions
    results["image_hashes"] = image_hashes
    results["image_embeddings"] = image_embeddings
    results["volume_items"] = volume_items

    return results


@click.command("filter-dataset")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=ANALYSIS_OUTPUT_DIR,
    help="Output directory for filtered dataset",
)
@click.option(
    "--output-format",
    type=click.Choice(["jsonl", "json", "parquet"]),
    default="jsonl",
    help="Output format",
)
@click.option(
    "--shard-size",
    type=int,
    default=5000,
    help="Number of rows per parquet shard (default: 5000, ~4GB each)",
)
@click.option(
    "--dummy",
    is_flag=True,
    help="Run in dummy mode with only 10 rows to check formatting",
)
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
    "--run-moderation/--no-moderation",
    default=False,
    help="Run OpenAI moderation on captions (default: enabled)",
)
@click.option(
    "--upload-r2/--no-upload-r2",
    default=True,
    help="Upload parquet shards to R2 bucket (default: enabled)",
)
@click.option(
    "--r2-prefix",
    type=str,
    default=None,
    help="Key prefix for R2 uploads (default: filtered_dataset_{DATETIME_SLUG})",
)
@click.option(
    "--sample",
    is_flag=True,
    help="Upload only the first 10,000 crops as a sample",
)
@click.option(
    "--prepare",
    is_flag=True,
    help="Prepare mode: run dedupe + selection, save manifest for parallel partition runs",
)
@click.option(
    "--manifest",
    type=click.Path(exists=True),
    default=None,
    help="Path to manifest JSON from --prepare run. Used with --partition.",
)
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Partition spec K/N (e.g. 1/4). Process every Nth item starting at K.",
)
def filter_dataset(
    output_dir,
    output_format,
    shard_size,
    dummy,
    detection_threshold,
    classification_threshold,
    run_moderation,
    upload_r2,
    r2_prefix,
    sample,
    prepare,
    manifest,
    partition,
):
    """
    Filter the dataset with confidence thresholds and deduplication.

    Applies the following filters:
    - Only detections with confidence >= detection_threshold (default 0.75)
    - Classifications with confidence < classification_threshold (default 0.70) are labeled "other"
    - Selects representative items from unique dedupe groups (intersection of hash and embedding groups)
    - Runs OpenAI moderation on captions (can be disabled with --no-moderation)

    By default, outputs parquet shards and uploads them to the R2 FILTER bucket in parallel.

    Parallel mode (run on N machines / tmux panes):
        1) filter-dataset --prepare                  # saves manifest.json
        2) filter-dataset --manifest manifest.json --partition 1/4
           filter-dataset --manifest manifest.json --partition 2/4
           filter-dataset --manifest manifest.json --partition 3/4
           filter-dataset --manifest manifest.json --partition 4/4

    Examples:
        filter-dataset
        filter-dataset --dummy
        filter-dataset --output-format json --no-upload-r2
        filter-dataset --output-format parquet
        filter-dataset --output-format parquet --shard-size 100000
        filter-dataset --detection-threshold 0.8 --classification-threshold 0.6
        filter-dataset --no-moderation
        filter-dataset --sample
        filter-dataset --r2-prefix my_export/v2
    """
    SAMPLE_LIMIT = 10_000

    # Validate partition arg
    partition_k, partition_n = None, None
    if partition:
        try:
            partition_k, partition_n = [int(x) for x in partition.split("/")]
            assert 1 <= partition_k <= partition_n
        except Exception:
            logger.error(f"Invalid --partition format '{partition}'. Expected K/N (e.g. 1/4)")
            return
    if partition and not manifest:
        logger.error("--partition requires --manifest from a --prepare run")
        return
    if prepare and manifest:
        logger.error("--prepare and --manifest are mutually exclusive")
        return

    logger.info("Starting dataset filtering...")
    logger.info(f"  Detection confidence threshold: {detection_threshold}")
    logger.info(f"  Classification confidence threshold: {classification_threshold}")
    logger.info(f"  Run moderation: {run_moderation}")
    logger.info(f"  Dummy mode: {dummy}")
    logger.info(f"  Upload to R2: {upload_r2}")
    logger.info(f"  Sample mode: {sample}")
    logger.info(f"  CPUS_LIMIT: {CPUS_LIMIT}")
    if prepare:
        logger.info("  Mode: PREPARE (will save manifest and exit)")
    if partition:
        logger.info(f"  Mode: PARTITION {partition_k}/{partition_n}")

    # ---- PARTITION MODE: load slim manifest, re-fetch DB data ----
    if manifest:
        logger.info(f"Loading manifest from {manifest}...")
        with open(manifest, "rb") as f:
            manifest_data = orjson.loads(f.read())

        dateslug = manifest_data.get("dateslug")
        if not dateslug:
            manifest_dir_name = Path(manifest).resolve().parent.name
            prefix = "filtered_dataset_"
            if manifest_dir_name.startswith(prefix):
                dateslug = manifest_dir_name[len(prefix):]
            else:
                dateslug = DATETIME_SLUG
    else:
        dateslug = DATETIME_SLUG

    # Create output directory
    output_path = Path(output_dir) / f"filtered_dataset_{dateslug}"
    output_path.mkdir(parents=True, exist_ok=True)

    if manifest:

        dets_by_item_all: dict[int, list[int]] = {int(k): v for k, v in manifest_data["dets_by_item"].items()}

        # Take this partition's slice of items FIRST, before building detections_data
        all_item_ids = sorted(dets_by_item_all.keys())
        if partition_k is not None and partition_n is not None:
            my_item_ids = [iid for i, iid in enumerate(all_item_ids) if (i % partition_n) == (partition_k - 1)]
        else:
            my_item_ids = all_item_ids

        dets_by_item: dict[int, list[int]] = {iid: dets_by_item_all[iid] for iid in my_item_ids}
        del dets_by_item_all
        my_det_id_set = set()
        for dids in dets_by_item.values():
            my_det_id_set.update(dids)

        # Only keep detections_data for this partition's detections
        detections_data = {}
        for k, v in manifest_data["detections_data"].items():
            int_k = int(k)
            if int_k in my_det_id_set:
                detections_data[int_k] = v
        del manifest_data

        my_detection_ids = list(my_det_id_set)
        del my_det_id_set
        logger.info(f"  Partition has {len(dets_by_item)} items, {len(my_detection_ids)} detections")

        # Re-fetch DB data sequentially to avoid holding all raw rows in memory at once
        logger.info(f"Fetching DB data sequentially for {len(my_detection_ids)} detections...")

        logger.info("  Fetching classifications...")
        classifications_by_det = {}
        for cls in _batched_iter(Classification, Classification.detection, my_detection_ids):
            pred_class = cls.pred_class
            pred_conf = cls.pred_conf
            if pred_conf is not None and pred_conf < classification_threshold:
                pred_class = "Other"
            classifications_by_det[cls.detection_id] = {
                "pred_class": pred_class,
                "pred_class_label": CLASSIFICATION_CLASS_DICT.get(pred_class, pred_class),
                "pred_conf": pred_conf,
                "probs": [float(p) for p in cls.probs] if cls.probs else None,
            }
        logger.info(f"    {len(classifications_by_det)} classifications")

        logger.info("  Fetching captions...")
        captions_by_det = {}
        for cap in _batched_iter(Caption, Caption.detection, my_detection_ids):
            captions_by_det[cap.detection_id] = {
                "text": cap.text,
                "lang": cap.lang,
                "lang_detected": cap.lang_detected,
                "linear_prob": cap.linear_prob,
                "thesaurus_matches": cap.thesaurus_matches,
            }
        logger.info(f"    {len(captions_by_det)} captions")

        logger.info("  Fetching volume barcodes...")
        item_ids = list(dets_by_item.keys())
        item_volumes = {}
        for item in _batched_iter(PipelineBatchItem, PipelineBatchItem.id_pipeline_batch_item, item_ids):
            item_volumes[item.id_pipeline_batch_item] = item.ib_volume_id
        logger.info(f"    {len(item_volumes)} volumes")

        logger.info("  Fetching image hashes...")
        image_hash_by_det = {}
        for ih in _batched_iter(ImageHash, ImageHash.detection, my_detection_ids):
            image_hash_by_det[ih.detection_id] = {"image_hash": ih.image_hash}
        logger.info(f"    {len(image_hash_by_det)} hashes")

        image_embedding_by_det = None
        del my_detection_ids
        logger.info("  Embeddings will be fetched per-shard to avoid OOM")
    else:
        # ---- NORMAL / PREPARE MODE: fetch from DB ----
        # Step 1: Get all detections with confidence >= threshold
        limit = None
        if dummy:
            limit = 1000
        elif sample:
            limit = SAMPLE_LIMIT * 3

        logger.info(f"Fetching detections with confidence >= {detection_threshold} (raw SQL, server-side cursor)...")
        detections_data = _raw_fetch_detections(detection_threshold, limit=limit)
        logger.info(f"  Found {len(detections_data)} detections meeting confidence threshold")

        if not detections_data:
            logger.warning("No detections found meeting the threshold")
            return

        detection_ids = list(detections_data.keys())

        # Step 2: Skip heavy DB fetch in prepare mode (partitions re-fetch their own data)
        if not prepare:
            db_data = _fetch_db_data_parallel(detection_ids, detections_data, CPUS_LIMIT)

            classifications_by_det = {}
            for cls in db_data["classifications"]:
                pred_class = cls.pred_class
                pred_conf = cls.pred_conf

                if pred_conf is not None and pred_conf < classification_threshold:
                    pred_class = "Other"

                classifications_by_det[cls.detection_id] = {
                    "pred_class": pred_class,
                    "pred_class_label": CLASSIFICATION_CLASS_DICT.get(pred_class, pred_class),
                    "pred_conf": pred_conf,
                    "probs": [float(p) for p in cls.probs] if cls.probs else None,
                }

            captions_by_det = {}
            for cap in db_data["captions"]:
                captions_by_det[cap.detection_id] = {
                    "text": cap.text,
                    "lang": cap.lang,
                    "lang_detected": cap.lang_detected,
                    "linear_prob": cap.linear_prob,
                    "thesaurus_matches": cap.thesaurus_matches,
                }

            item_volumes = {}
            for item in db_data["volume_items"]:
                item_volumes[item.id_pipeline_batch_item] = item.ib_volume_id

            image_hash_by_det = {}
            for ih in db_data["image_hashes"]:
                image_hash_by_det[ih.detection_id] = {
                    "id_imagehash": ih.id_imagehash,
                    "image_hash": ih.image_hash,
                    "created": ih.created.isoformat() if ih.created else None,
                }

            image_embedding_by_det = {}
            for ie in db_data["image_embeddings"]:
                embedding_vector = [float(x) for x in ie.embedding] if ie.embedding is not None else None
                image_embedding_by_det[ie.detection_id] = {
                    "id_embedding": ie.id_embedding,
                    "embedding": embedding_vector,
                    "created": ie.created.isoformat() if ie.created else None,
                }

            del db_data

        # Step 3: Get dedupe intersection groups (use raw SQL for prepare mode)
        logger.info("Computing dedupe intersection groups...")
        if prepare:
            logger.info("  Using SQL JOIN for dedupe intersection (single pass)...")
            intersection_groups = _raw_fetch_dedupe_intersection()
        else:
            intersection_groups = get_dedupe_intersection_groups(detection_ids)

        # Step 4: Select representatives from each intersection group
        import time as _time
        logger.info(f"Selecting representatives from {len(intersection_groups):,} dedupe groups...")
        selected_detection_ids = set()

        detections_in_groups = set()
        t0_sel = _time.time()
        t_last_sel = t0_sel
        groups_processed = 0
        total_groups = len(intersection_groups)
        for group_key, group_det_ids in intersection_groups.items():
            valid_det_ids = [d for d in group_det_ids if d in detections_data]
            if valid_det_ids:
                representative = select_representative(valid_det_ids, detections_data)
                selected_detection_ids.add(representative)
                detections_in_groups.update(valid_det_ids)
            groups_processed += 1
            now = _time.time()
            if now - t_last_sel >= 15:
                logger.info(f"    ... {groups_processed:,}/{total_groups:,} groups ({groups_processed*100//total_groups}%, {now - t0_sel:.0f}s elapsed)")
                t_last_sel = now

        detections_without_group = set(detections_data.keys()) - detections_in_groups
        selected_detection_ids.update(detections_without_group)

        logger.info(f"  Detections in dedupe groups: {len(detections_in_groups)}")
        logger.info(f"  Detections without groups (unique): {len(detections_without_group)}")
        logger.info(f"  Total selected (after deduplication): {len(selected_detection_ids)}")

        # Group selected detections by pipeline_batch_item for efficient scan loading
        dets_by_item: dict[int, list[int]] = defaultdict(list)
        for det_id in selected_detection_ids:
            item_id = detections_data[det_id]["pipeline_batch_item_id"]
            dets_by_item[item_id].append(det_id)

        # ---- PREPARE MODE: save slim manifest and exit ----
        if prepare:
            manifest_path = output_path / "manifest.json"
            logger.info(f"Building manifest dict ({len(detections_data):,} detections, {len(dets_by_item):,} items)...")
            manifest_out = {
                "detection_threshold": detection_threshold,
                "classification_threshold": classification_threshold,
                "detections_data": {str(k): v for k, v in detections_data.items()},
                "dets_by_item": {str(k): v for k, v in dets_by_item.items()},
                "total_selected": len(selected_detection_ids),
                "total_items": len(dets_by_item),
                "dateslug": DATETIME_SLUG,
            }
            logger.info("Serializing manifest with orjson...")
            with open(manifest_path, "wb") as f:
                f.write(orjson.dumps(manifest_out))
            manifest_mb = manifest_path.stat().st_size / 1024 / 1024
            logger.success(f"Manifest saved to {manifest_path} ({manifest_mb:.0f} MB, {len(selected_detection_ids)} detections across {len(dets_by_item)} items)")
            logger.info(f"Run partitions with: --manifest {manifest_path} --partition K/N")

            if upload_r2:
                s3_client = get_s3_client("FILTER")
                manifest_s3_key = f"{DATETIME_SLUG}-manifest.json"
                logger.info(f"Uploading manifest to R2: {manifest_s3_key}...")
                try:
                    with open(manifest_path, "rb") as fh:
                        s3_client.put_object(
                            Bucket=FILTER_STORAGE_BUCKET_NAME,
                            Key=manifest_s3_key,
                            Body=fh,
                            ContentType="application/json",
                        )
                    logger.success(f"Manifest uploaded to R2: {manifest_s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload manifest to R2: {e}")

            return

    # Step 5: Build filtered dataset, streaming to shards
    logger.info("Building filtered dataset (streaming to shards)...")

    moderation_client = None
    if run_moderation:
        logger.info("Initializing OpenAI client for moderation...")
        moderation_client = openai.OpenAI(timeout=OPENAI_REQUEST_TIMEOUT)

    total_items = len(dets_by_item)

    # Apply sample limit
    record_limit = None
    if dummy:
        record_limit = 100
    elif sample:
        record_limit = SAMPLE_LIMIT

    effective_shard_size = shard_size or 5000

    # R2 upload setup
    s3_client = get_s3_client("FILTER") if upload_r2 else None
    bucket_name = FILTER_STORAGE_BUCKET_NAME if upload_r2 else None
    upload_executor = ThreadPoolExecutor(max_workers=10) if upload_r2 else None
    upload_futures = []
    upload_stats = {"uploaded": 0, "failed": 0}
    MAX_PENDING_UPLOADS = 8

    # Check which shards already exist in R2 to allow resuming
    existing_shards: set[str] = set()
    if upload_r2 and s3_client:
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=dateslug):
                for obj in page.get("Contents", []):
                    existing_shards.add(obj["Key"])
            if existing_shards:
                logger.info(f"  Found {len(existing_shards)} existing shards in R2, will skip them")
        except Exception as e:
            logger.warning(f"  Could not list existing R2 shards: {e}")

    # Fast-forward past records covered by existing shards
    records_to_skip = len(existing_shards) * effective_shard_size if existing_shards else 0
    if records_to_skip:
        logger.info(f"  Will skip first {records_to_skip} records ({len(existing_shards)} existing shards x {effective_shard_size} rows)")

    # Streaming state — records are written in small row groups (ROW_GROUP_SIZE)
    # to an open ParquetWriter so crop bytes never accumulate past ~500 rows.
    row_group_buf: list[dict] = []
    row_group_crop_bytes: int = 0  # track crop_gen column size to flush before 2GB Parquet limit
    shard_idx = len(existing_shards)
    shard_record_count = 0  # records written into the current shard so far
    total_records = 0
    skipped_records = 0
    class_counts = defaultdict(int)
    reclassified_count = 0
    thesaurus_match_count = 0

    shard_prefix = dateslug

    # Current open shard writer state (managed by _open_shard / _flush_row_group / _close_shard)
    _current_writer: pq.ParquetWriter | None = None
    _current_tmp_path: str | None = None
    _current_s3_key: str | None = None
    # IDs in the current shard, used for just-in-time embedding fetch
    _current_shard_ids: list[int] = []

    def _open_shard():
        nonlocal shard_idx, shard_record_count, _current_writer, _current_tmp_path, _current_s3_key, _current_shard_ids
        shard_idx += 1
        _current_s3_key = f"{shard_prefix}-{shard_idx:04d}.parquet"
        tmp_fd, _current_tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(tmp_fd)
        _current_writer = pq.ParquetWriter(_current_tmp_path, PARQUET_SCHEMA)
        shard_record_count = 0
        _current_shard_ids = []

    def _flush_row_group():
        nonlocal row_group_buf, shard_record_count, row_group_crop_bytes
        if not row_group_buf:
            return

        # Just-in-time embedding fetch for this row group
        if image_embedding_by_det is None:
            rg_det_ids = [r["id"] for r in row_group_buf]
            rg_embeddings = {}
            for ie in _batched_iter(ImageEmbedding, ImageEmbedding.detection, rg_det_ids):
                rg_embeddings[ie.detection_id] = [float(x) for x in ie.embedding] if ie.embedding is not None else None
            for r in row_group_buf:
                r["embedding_gen"] = rg_embeddings.get(r["id"])
            del rg_embeddings

        _current_writer.write_table(_build_row_group_table(row_group_buf))
        shard_record_count += len(row_group_buf)
        row_group_buf = []
        row_group_crop_bytes = 0

    def _close_shard():
        nonlocal _current_writer, _current_tmp_path, _current_s3_key, _current_shard_ids
        if _current_writer is None:
            return

        _flush_row_group()
        _current_writer.close()
        _current_writer = None
        _current_shard_ids = []

        tmp_size_mb = os.path.getsize(_current_tmp_path) / 1024 / 1024

        if upload_r2:
            # Drain completed futures to free resources
            still_pending = []
            for f, key in upload_futures:
                if f.done():
                    try:
                        if f.result():
                            upload_stats["uploaded"] += 1
                        else:
                            upload_stats["failed"] += 1
                            logger.error(f"  Upload failed for {key}")
                    except Exception as e:
                        upload_stats["failed"] += 1
                        logger.error(f"  Upload exception for {key}: {e}")
                else:
                    still_pending.append((f, key))
            upload_futures[:] = still_pending

            # Backpressure: wait until pending uploads drop below threshold
            while len(upload_futures) >= MAX_PENDING_UPLOADS:
                logger.info(f"  Upload backpressure: {len(upload_futures)} uploads pending, waiting...")
                for f, _ in upload_futures:
                    if not f.done():
                        f.result(timeout=30)
                        break
                upload_futures[:] = [(f, k) for f, k in upload_futures if not f.done()]

            logger.info(f"  Closed shard {shard_idx} ({shard_record_count} rows, {tmp_size_mb:.0f} MB) -> uploading to R2")
            fut = upload_executor.submit(_upload_parquet_file_to_r2, s3_client, _current_tmp_path, _current_s3_key, bucket_name)
            upload_futures.append((fut, _current_s3_key))
        else:
            shard_file = output_path / _current_s3_key
            os.rename(_current_tmp_path, str(shard_file))
            logger.info(f"  Wrote shard {shard_idx}: {shard_file} ({shard_record_count} rows)")

        _current_tmp_path = None
        _current_s3_key = None

    MAX_INFLIGHT = 12
    image_load_workers = MAX_INFLIGHT
    item_list = list(dets_by_item.items())
    done = False
    items_processed = 0

    logger.info(f"  Processing {total_items} items with {image_load_workers} workers (max {MAX_INFLIGHT} in-flight)...")
    logger.info(f"  Shard size: {effective_shard_size} rows")

    def _load_and_crop(item_id: int, item_det_ids: list[int]) -> tuple[int, list[int], dict[int, bytes | None]]:
        """Load scan images one at a time, crop, then free the decoded array before the next."""
        try:
            item = PipelineBatchItem.get(PipelineBatchItem.id_pipeline_batch_item == item_id)
            all_images = item.data.images
        except Exception:
            return (item_id, item_det_ids, {})

        # Group detections by scan filename so we decode each scan only once
        dets_by_scan: dict[str, list[int]] = defaultdict(list)
        for det_id in item_det_ids:
            dets_by_scan[detections_data[det_id]["scan_filename"]].append(det_id)

        crops: dict[int, bytes | None] = {}
        for scan_fn, det_ids_for_scan in dets_by_scan.items():
            key = next((k for k in all_images if str(k) == scan_fn), None)
            if key is None:
                for det_id in det_ids_for_scan:
                    crops[det_id] = None
                continue
            try:
                decoded = decode_image_bytes(all_images[key])
            except Exception:
                for det_id in det_ids_for_scan:
                    crops[det_id] = None
                continue

            for det_id in det_ids_for_scan:
                det = detections_data[det_id]
                if det["bbox_xyxy"]:
                    crops[det_id] = generate_crop_png_bytes(det["bbox_xyxy"], decoded)
                else:
                    crops[det_id] = None
            del decoded

        del all_images, item
        return (item_id, item_det_ids, crops)

    # Fast-forward: skip entire items whose detections fall within already-uploaded shards
    skip_item_list = []
    process_item_list = []
    if records_to_skip:
        cumulative = 0
        for item_id, det_ids in item_list:
            if cumulative + len(det_ids) <= records_to_skip:
                cumulative += len(det_ids)
                skip_item_list.append((item_id, det_ids))
            else:
                process_item_list.append((item_id, det_ids))
        skipped_records = cumulative
        logger.info(f"  Fast-forwarding past {len(skip_item_list)} items ({skipped_records} records) covered by existing shards")
        items_processed = len(skip_item_list)
    else:
        process_item_list = item_list

    with ThreadPoolExecutor(max_workers=image_load_workers) as executor:
        pending = {}
        item_iter = iter(enumerate(process_item_list))

        def _submit_next():
            try:
                idx, (item_id, det_ids) = next(item_iter)
                f = executor.submit(_load_and_crop, item_id, det_ids)
                pending[f] = item_id
            except StopIteration:
                pass

        for _ in range(MAX_INFLIGHT):
            _submit_next()

        while pending and not done:
            finished = next(as_completed(pending))
            del pending[finished]
            _submit_next()

            item_id, item_det_ids, crops = finished.result()
            items_processed += 1

            if items_processed % 50 == 0 or items_processed == total_items:
                logger.info(f"  Processed {items_processed}/{total_items} items ({total_records} records, shard {shard_idx})")

            for det_id in item_det_ids:
                if done:
                    break

                # Skip individual records in the first partially-covered item
                if skipped_records < records_to_skip:
                    skipped_records += 1
                    continue

                det_data = detections_data[det_id]
                cls_data = classifications_by_det.get(det_id, {})
                cap_data = captions_by_det.get(det_id, {})
                hash_data = image_hash_by_det.get(det_id, {})
                embedding_data = image_embedding_by_det.get(det_id, {}) if image_embedding_by_det is not None else {}

                volume_barcode = item_volumes.get(det_data["pipeline_batch_item_id"])

                crop_bytes = crops.get(det_id)

                bbox_xywh = det_data["bbox_xywh"]
                if bbox_xywh and len(bbox_xywh) >= 4:
                    width = bbox_xywh[2]
                    height = bbox_xywh[3]
                    pixel_count_mpx = (width * height) / 1_000_000
                else:
                    width = None
                    height = None
                    pixel_count_mpx = None

                classification_label = cls_data.get("pred_class_label") if cls_data else None
                is_non_captionable = classification_label in ("Artifact", "Ex Libris/Decorative")

                if cap_data and cap_data.get("text"):
                    caption_text = cap_data.get("text")
                    if caption_text in ("Undetermined", "Undetermined."):
                        caption_text = "CAPTION FAILED"
                    caption_lang_passed = lang_name_to_iso639_3(cap_data.get("lang"))
                elif is_non_captionable:
                    caption_text = None
                    caption_lang_passed = None
                else:
                    caption_text = "CAPTION FAILED"
                    caption_lang_passed = None

                moderation_result = None
                if run_moderation and cap_data and cap_data.get("text"):
                    moderation_result = run_moderation_fn(moderation_client, cap_data.get("text"))

                caption_is_valid = caption_text is not None and caption_text != "CAPTION FAILED"
                caption_linear_prob = cap_data.get("linear_prob") if cap_data and caption_is_valid else None
                caption_lang_detected = cap_data.get("lang_detected") if cap_data and caption_is_valid else None
                thesaurus_matches = cap_data.get("thesaurus_matches") if cap_data and caption_is_valid else None
                if isinstance(thesaurus_matches, str) and thesaurus_matches == "null":
                    thesaurus_matches = None
                if not caption_is_valid:
                    caption_lang_passed = None

                classification_probs_formatted = format_classification_probs(
                    cls_data.get("probs") if cls_data else None
                )

                record_id = generate_record_id(det_id)

                record = {
                    "id": record_id,
                    "crop_gen": crop_bytes,
                    "barcode_src": volume_barcode,
                    "page_filename_src": det_data["scan_filename"],
                    "bbox_xyxy_gen": det_data["bbox_xyxy"],
                    "width_gen": width,
                    "height_gen": height,
                    "pixel_count_mpx_gen": pixel_count_mpx,
                    "detection_confidence_gen": det_data["bbox_conf"],
                    "classification_gen": classification_label,
                    "classification_confidence_gen": cls_data.get("pred_conf") if cls_data else None,
                    "classification_probs_gen": classification_probs_formatted,
                    "phash_gen": hash_data.get("image_hash") if hash_data else None,
                    "embedding_gen": embedding_data.get("embedding") if embedding_data else None,
                    "caption_exp": caption_text,
                    "caption_linear_prob_exp": caption_linear_prob,
                    "caption_lang_passed_exp": caption_lang_passed,
                    "caption_lang_detected_exp": caption_lang_detected,
                    "caption_chronam_thesauri_matches_exp": thesaurus_matches,
                }

                # Open a new shard writer if needed
                if _current_writer is None:
                    _open_shard()

                row_group_buf.append(record)
                row_group_crop_bytes += len(crop_bytes) if crop_bytes else 0
                total_records += 1

                # Incremental stats
                cls_label = classification_label or "Unknown"
                class_counts[cls_label] += 1
                if (
                    classification_label == "Other"
                    and cls_data.get("pred_conf") is not None
                    and cls_data["pred_conf"] < classification_threshold
                ):
                    reclassified_count += 1
                if thesaurus_matches:
                    thesaurus_match_count += 1

                # Flush row group when it reaches ROW_GROUP_SIZE or crop column approaches 2GB
                if len(row_group_buf) >= ROW_GROUP_SIZE or row_group_crop_bytes >= 1_900_000_000:
                    _flush_row_group()

                # Close shard when it reaches effective_shard_size
                if shard_record_count + len(row_group_buf) >= effective_shard_size:
                    _close_shard()

                if record_limit and total_records >= record_limit:
                    done = True

            del crops

    # Close any open shard with remaining records
    if _current_writer is not None:
        _close_shard()
    elif row_group_buf:
        _open_shard()
        _close_shard()

    logger.info(f"Total records: {total_records} across {shard_idx} shards")

    # Wait for remaining uploads to finish
    if upload_r2:
        logger.info("Waiting for remaining R2 uploads to complete...")
        for f, s3_key in upload_futures:
            try:
                if f.result():
                    upload_stats["uploaded"] += 1
                else:
                    upload_stats["failed"] += 1
            except Exception as e:
                upload_stats["failed"] += 1
                logger.error(f"  Upload exception for {s3_key}: {e}")
        upload_futures.clear()
        upload_executor.shutdown(wait=False)
        logger.info(f"  R2 upload complete: {upload_stats['uploaded']} succeeded, {upload_stats['failed']} failed")

    # Write stats summary
    stats_suffix = "" if partition_k else "" # do we need partition name?
    stats = {
        "detection_threshold": detection_threshold,
        "classification_threshold": classification_threshold,
        "moderation_enabled": run_moderation,
        "partition": f"{partition_k}/{partition_n}" if partition_k else None,
        "final_record_count": total_records,
        "total_shards": shard_idx,
        "shard_size": effective_shard_size,
        "dummy_mode": dummy,
        "sample_mode": sample,
        "uploaded_to_r2": upload_r2,
        "classification_distribution": dict(class_counts),
        "reclassified_to_other": reclassified_count,
        "thesaurus_match_count": thesaurus_match_count,
    }

    stats_file = output_path / f"filter_stats{stats_suffix}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to: {stats_file}")

    # Print summary
    logger.info("=" * 60)
    logger.info(f"FILTERING SUMMARY{f' (partition {partition_k}/{partition_n})' if partition_k else ''}")
    logger.info("=" * 60)
    logger.info(f"Detection threshold: >= {detection_threshold}")
    logger.info(f"Classification threshold: < {classification_threshold} -> 'Other'")
    logger.info(f"After deduplication: {total_records}")
    logger.info(f"Total shards: {shard_idx} ({effective_shard_size} rows each)")
    logger.info(f"Reclassified to 'Other': {reclassified_count}")
    if sample:
        logger.info(f"Sample mode: limited to first {SAMPLE_LIMIT} crops")
    logger.info("-" * 60)
    logger.info("Classification distribution:")
    for cls_label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls_label}: {count}")
    logger.info("=" * 60)

    logger.success(f"Filtering complete! Output: {output_path}")
