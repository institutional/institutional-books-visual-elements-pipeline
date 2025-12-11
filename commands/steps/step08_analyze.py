import csv
import json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool

import click
from loguru import logger

from const import CPUS_LIMIT, ANALYSIS_OUTPUT_DIR
from utils import get_db
from models import (
    PipelineRun,
    PipelineBatch,
    PipelineBatchItem,
    IBVolume,
    Detection,
    DedupedHash,
    DedupedEmbedding,
)


def calculate_crop_dimensions(bbox_xywh):
    """Calculate width and height from bbox_xywh."""
    if bbox_xywh is None or len(bbox_xywh) != 4:
        return None, None
    return bbox_xywh[2], bbox_xywh[3]  # width, height


def calculate_crop_area(bbox_xywh):
    """Calculate area from bbox_xywh."""
    width, height = calculate_crop_dimensions(bbox_xywh)
    if width is None or height is None:
        return None
    return width * height


def process_detection(args):
    """
    Process a single detection and return its row data.
    Must be a module-level function for multiprocessing pickling.
    """
    detection_id, hash_groups, hash_group_sizes, embedding_groups, embedding_group_sizes = args

    # Ensure fresh database connection for this worker process
    db = get_db()
    if db.is_closed():
        db.connect(reuse_if_open=True)

    try:
        # Fetch detection with all related data
        detection = (
            Detection.select(Detection, PipelineBatchItem, IBVolume, PipelineBatch)
            .join(
                PipelineBatchItem,
                on=(Detection.pipeline_batch_item == PipelineBatchItem.id_pipeline_batch_item),
            )
            .join(
                PipelineBatch,
                on=(PipelineBatchItem.pipeline_batch == PipelineBatch.id_pipeline_batch),
            )
            .switch(PipelineBatchItem)
            .join(IBVolume, on=(PipelineBatchItem.ib_volume == IBVolume.barcode))
            .where(Detection.id_detection == detection_id)
            .get()
        )

        # Basic IDs
        row = {
            "id_detection": detection.id_detection,
            "id_pipeline_batch_item": detection.pipeline_batch_item.id_pipeline_batch_item,
            "id_pipeline_batch": detection.pipeline_batch_item.pipeline_batch.id_pipeline_batch,
            "barcode": detection.pipeline_batch_item.ib_volume.barcode,
            "scan_filename": detection.scan_filename,
        }

        # Volume metadata
        metadata = detection.pipeline_batch_item.ib_volume.metadata
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        elif metadata is None:
            metadata = {}
        row["topic"] = metadata.get("topic_or_subject_src")
        row["date_published"] = metadata.get("date1_src")
        row["language"] = metadata.get("language_src")
        row["title"] = metadata.get("title_src")
        row["author"] = metadata.get("author_src")

        # Crop dimensions
        width, height = calculate_crop_dimensions(detection.bbox_xywh)
        row["crop_width"] = width
        row["crop_height"] = height
        row["crop_area"] = calculate_crop_area(detection.bbox_xywh)
        row["bbox_conf"] = detection.bbox_conf

        # Classification data
        classifications = list(detection.classifications)
        if classifications:
            # Get the primary classification (highest confidence)
            primary_class = max(classifications, key=lambda c: c.pred_conf)
            row["pred_class"] = primary_class.pred_class
            row["pred_conf"] = primary_class.pred_conf
            row["pred_idx"] = primary_class.pred_idx
            row["num_classifications"] = len(classifications)
        else:
            row["pred_class"] = None
            row["pred_conf"] = None
            row["pred_idx"] = None
            row["num_classifications"] = 0

        # Caption data
        captions = list(detection.captions)
        if captions:
            # Get primary caption (first one)
            primary_caption = captions[0]
            caption_text = primary_caption.caption
            row["caption"] = caption_text
            row["caption_length"] = len(caption_text)
            row["caption_word_count"] = len(caption_text.split())
            row["caption_lang"] = primary_caption.lang
            row["num_captions"] = len(captions)
        else:
            row["caption"] = None
            row["caption_length"] = None
            row["caption_word_count"] = None
            row["caption_lang"] = None
            row["num_captions"] = 0

        # Deduplication data
        # Hash-based deduplication
        if detection_id in hash_groups:
            hash_group_id = hash_groups[detection_id]
            row["hash_group_id"] = hash_group_id
            row["hash_group_size"] = hash_group_sizes[hash_group_id]
            row["is_hash_duplicate"] = hash_group_sizes[hash_group_id] > 1
        else:
            row["hash_group_id"] = None
            row["hash_group_size"] = None
            row["is_hash_duplicate"] = None

        # Embedding-based deduplication
        if detection_id in embedding_groups:
            emb_group_id = embedding_groups[detection_id]
            row["embedding_group_id"] = emb_group_id
            row["embedding_group_size"] = embedding_group_sizes[emb_group_id]
            row["is_embedding_duplicate"] = embedding_group_sizes[emb_group_id] > 1
        else:
            row["embedding_group_id"] = None
            row["embedding_group_size"] = None
            row["is_embedding_duplicate"] = None

        return row

    except Exception as e:
        logger.error(f"Error processing detection {detection_id}: {e}")
        return None


@click.command("step08-analyze")
@click.option("--id-pipeline-run", type=int, required=True, help="Pipeline run to analyze")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=ANALYSIS_OUTPUT_DIR,
    help="Output directory for analysis files",
)
def step08_analyze(id_pipeline_run, output_dir):
    """
    Collect analysis data from a pipeline run and export to CSV for further analysis.

    Gathers crop-level metrics including metadata, dimensions, confidence scores,
    duplication information, and caption statistics.
    """
    logger.info(f"Starting analysis for pipeline run {id_pipeline_run}")

    # Get database connection
    db = get_db()

    # Validate pipeline run exists
    try:
        pipeline_run = PipelineRun.get_by_id(id_pipeline_run)
    except PipelineRun.DoesNotExist:
        logger.error(f"Pipeline run {id_pipeline_run} not found")
        return

    logger.info(
        f"Found pipeline run with {pipeline_run.items_total} items in {pipeline_run.batches_total} batches"
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare output file
    output_file = output_path / f"analysis_run_{id_pipeline_run}.csv"

    # Build deduplication lookup tables
    logger.info("Building deduplication lookup tables...")
    hash_groups = {}
    embedding_groups = {}

    for dh in DedupedHash.select():
        hash_groups[dh.detection_id] = dh.group_id

    for de in DedupedEmbedding.select():
        embedding_groups[de.detection_id] = de.group_id

    # Count group sizes for duplication rate
    hash_group_sizes = defaultdict(int)
    embedding_group_sizes = defaultdict(int)

    for group_id in hash_groups.values():
        hash_group_sizes[group_id] += 1

    for group_id in embedding_groups.values():
        embedding_group_sizes[group_id] += 1

    logger.info(f"Found {len(hash_groups)} hash entries in {len(hash_group_sizes)} groups")
    logger.info(
        f"Found {len(embedding_groups)} embedding entries in {len(embedding_group_sizes)} groups"
    )

    # Get all batches for this pipeline run - MATERIALIZE THE LIST
    batches = list(
        PipelineBatch.select()
        .where(PipelineBatch.pipeline_run == id_pipeline_run)
        .order_by(PipelineBatch.id_pipeline_batch)
    )

    all_rows = []

    logger.info(f"Processing {len(batches)} batches using {CPUS_LIMIT} workers...")

    # Process each batch
    for batch_idx, batch in enumerate(batches, 1):
        # Get all detection IDs for this batch - MATERIALIZE THE LIST
        detection_ids = [
            d.id_detection
            for d in Detection.select(Detection.id_detection)
            .join(
                PipelineBatchItem,
                on=(Detection.pipeline_batch_item == PipelineBatchItem.id_pipeline_batch_item),
            )
            .where(PipelineBatchItem.pipeline_batch == batch.id_pipeline_batch)
        ]

        if not detection_ids:
            logger.warning(
                f"Batch {batch_idx}/{len(batches)} (ID: {batch.id_pipeline_batch}): No detections found"
            )
            continue

        # Close connection before forking to prevent connection sharing
        db.close()

        # Prepare arguments for multiprocessing
        args_list = [
            (det_id, hash_groups, hash_group_sizes, embedding_groups, embedding_group_sizes)
            for det_id in detection_ids
        ]

        # Process detections in parallel
        with Pool(processes=CPUS_LIMIT) as pool:
            batch_rows = pool.map(process_detection, args_list)

        # Reconnect after pool is done
        if db.is_closed():
            db.connect(reuse_if_open=True)

        # Filter out any None results from errors
        batch_rows = [row for row in batch_rows if row is not None]

        all_rows.extend(batch_rows)

        logger.info(
            f"Batch {batch_idx}/{len(batches)} (ID: {batch.id_pipeline_batch}) complete: "
            f"Processed {len(batch_rows)} detections (Total: {len(all_rows)})"
        )

    # Write to CSV
    logger.info(f"Writing {len(all_rows)} rows to {output_file}")

    if all_rows:
        fieldnames = all_rows[0].keys()

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        logger.success(f"Analysis complete! Data written to {output_file}")

        # Print summary statistics
        logger.info("Summary statistics:")
        logger.info(f"  Total detections: {len(all_rows)}")

        with_classifications = sum(1 for r in all_rows if r["pred_conf"] is not None)
        logger.info(
            f"  With classifications: {with_classifications} ({with_classifications/len(all_rows)*100:.1f}%)"
        )

        with_captions = sum(1 for r in all_rows if r["caption"] is not None)
        logger.info(f"  With captions: {with_captions} ({with_captions/len(all_rows)*100:.1f}%)")

        hash_duplicates = sum(1 for r in all_rows if r["is_hash_duplicate"])
        logger.info(
            f"  Hash duplicates: {hash_duplicates} ({hash_duplicates/len(all_rows)*100:.1f}%)"
        )

        emb_duplicates = sum(1 for r in all_rows if r["is_embedding_duplicate"])
        logger.info(
            f"  Embedding duplicates: {emb_duplicates} ({emb_duplicates/len(all_rows)*100:.1f}%)"
        )
    else:
        logger.warning("No data collected!")
