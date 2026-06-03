import click
from loguru import logger
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing

from models import PipelineBatch, Detection
from const import CLASSIFICATION_CLASS_DICT, PEEK_OUTPUT_DIR, DATETIME_SLUG

# General
JPEG_QUALITY = 95
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Bounding Box
BBOX_NUMBER_COLOR = (0, 0, 0)
BBOX_COLOR_CAPTION = (0, 255, 0)
BBOX_COLOR_NO_CAPTION = (0, 0, 255)

# Heading
HEADING_HEIGHT = 100
HEADING_BG_COLOR = (0, 0, 0)
HEADING_FONT_SCALE = 1.2
HEADING_FONT_COLOR = (255, 255, 255)
HEADING_WEIGHT = 3

# Subheading
SUBHEADING_FONT_SCALE = 0.8
SUBHEADING_FONT_COLOR = (200, 200, 200)
SUBHEADING_WEIGHT = 2

# Legend
LEGEND_BG_COLOR = (255, 255, 255)
LEGEND_PADDING = 20
LEGEND_FONT_SCALE = 0.7
LEGEND_LINE_SPACING = 35
DETECTION_NUMBER_FONT_COLOR = (0, 0, 0)
DETECTION_CAPTION_FONT_COLOR = (50, 50, 50)
DETECTION_NUMBER_WEIGHT = 3
DETECTION_CAPTION_WEIGHT = 1
MAX_CAPTION_WIDTH = 100


def _process_page_image(page_data: dict) -> dict:
    """
    Worker function to process a single page image.
    Takes raw data (not peewee models) and returns result info.
    """
    from textwrap import wrap

    page_idx = page_data["page_idx"]
    volume_barcode = page_data["volume_barcode"]
    scan_filename = page_data["scan_filename"]
    image_bytes = page_data["image_bytes"]
    detections = page_data["detections"]  # List of dicts with bbox, caption, classification info
    output_path = page_data["output_path"]

    try:
        # Decode full image
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        full_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

        # Draw numbered bounding boxes on the image
        image_with_boxes = full_image.copy()

        for idx, det in enumerate(detections, 1):
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Draw rectangle (green if caption exists, red if failed)
            has_caption = det["has_caption"]
            color = BBOX_COLOR_CAPTION if has_caption else BBOX_COLOR_NO_CAPTION
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 4)

            # Draw number in circle at top-left of bbox
            circle_center = (x1 + 30, y1 + 30)
            cv2.circle(image_with_boxes, circle_center, 25, color, -1)
            cv2.putText(
                image_with_boxes,
                str(idx),
                (x1 + 15, y1 + 40),
                FONT,
                1.0,
                BBOX_NUMBER_COLOR,
                3,
            )

        # Create legend panel
        font = FONT
        font_scale = LEGEND_FONT_SCALE
        line_spacing = LEGEND_LINE_SPACING
        padding = LEGEND_PADDING
        max_caption_width = MAX_CAPTION_WIDTH

        # Build legend lines
        legend_lines = []
        for idx, det in enumerate(detections, 1):
            class_info = ""
            detection_info = ""
            if det["class_name"]:
                detection_info = f" - Detection Confidence: {det['det_conf']}"
                class_info = f" - {det['class_name']} - Class Confidence: {det['class_conf']}"
            legend_lines.append(f"[{idx}]{detection_info}{class_info}")

            caption_text = det["caption"] if det["caption"] else "(NO CAPTION)"
            wrapped = wrap(caption_text, width=max_caption_width)
            legend_lines.extend([f"  {line}" for line in wrapped])
            legend_lines.append("")

        legend_height = padding * 2 + len(legend_lines) * line_spacing
        legend_width = image_with_boxes.shape[1]

        legend_panel = np.full(
            (legend_height, legend_width, 3), LEGEND_BG_COLOR, dtype=np.uint8
        )

        y_offset = padding + 25
        for line in legend_lines:
            if line.startswith("["):
                color = DETECTION_NUMBER_FONT_COLOR
                weight = DETECTION_NUMBER_WEIGHT
            else:
                color = DETECTION_CAPTION_FONT_COLOR
                weight = DETECTION_CAPTION_WEIGHT

            cv2.putText(
                legend_panel,
                line,
                (padding, y_offset),
                font,
                font_scale,
                color,
                weight,
            )
            y_offset += line_spacing

        # Create header
        header_height = HEADING_HEIGHT
        header = np.full((header_height, legend_width, 3), HEADING_BG_COLOR, dtype=np.uint8)

        header_text = f"{volume_barcode} | {scan_filename}"
        cv2.putText(
            header,
            header_text,
            (20, 35),
            FONT,
            HEADING_FONT_SCALE,
            HEADING_FONT_COLOR,
            HEADING_WEIGHT,
        )

        cls_success = sum(1 for d in detections if d["class_name"])
        cls_failed = len(detections) - cls_success
        cls_text = f"Classifications: {cls_success} | Failed: {cls_failed}"
        cv2.putText(
            header,
            cls_text,
            (20, 65),
            FONT,
            SUBHEADING_FONT_SCALE,
            SUBHEADING_FONT_COLOR,
            SUBHEADING_WEIGHT,
        )

        cap_success = sum(1 for d in detections if d["has_caption"])
        cap_failed = len(detections) - cap_success
        cap_text = f"Captions: {cap_success} | Failed: {cap_failed}"
        cv2.putText(
            header,
            cap_text,
            (20, 90),
            FONT,
            SUBHEADING_FONT_SCALE,
            SUBHEADING_FONT_COLOR,
            SUBHEADING_WEIGHT,
        )

        output_image = np.vstack([header, image_with_boxes, legend_panel])

        scan_name = scan_filename.replace(".jp2", "").replace(".tiff", "").replace(".tif", "")
        output_filename = Path(output_path) / f"{page_idx:04d}_{volume_barcode}_{scan_name}.jpg"
        cv2.imwrite(
            str(output_filename), output_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        return {
            "success": True,
            "page_idx": page_idx,
            "filename": output_filename.name,
            "num_detections": len(detections),
        }

    except Exception as e:
        return {
            "success": False,
            "page_idx": page_idx,
            "error": str(e),
        }


@click.command("peek")
@click.option("--scope", type=click.Choice(["detection", "deduplication"]), required=True)
@click.option("--id-pipeline-batch", type=int, required=True, help="Pipeline batch ID to inspect")
@click.option("--n", help="Number of random items to select (integer or 'all')")
@click.option(
    "--sample-type",
    type=click.Choice(["volumes", "pages"]),
    default="volumes",
    help="Sample random volumes or random pages",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=PEEK_OUTPUT_DIR,
    help="Output directory for visualization",
)
def peek(scope, id_pipeline_batch, n, sample_type, output_dir):
    """
    Peek at random samples to visually confirm the pipeline is working as expected.

    Supports both batch-level steps (detection, classification, captioning) and
    run-level steps (embedding, hash deduplication).

    Examples:
        peek --scope detection --id-pipeline-batch 123 --n 5
        peek --scope detection --id-pipeline-batch 123 --n 50 --sample-type pages
        peek --scope deduplication --id-pipeline-batch 123 --n all
    """

    # Validate pipeline batch exists
    try:
        batch = PipelineBatch.get_by_id(id_pipeline_batch)
    except Exception as e:
        logger.error(f"Pipeline batch {id_pipeline_batch} not found")
        click.get_current_context().exit(1)

    output_path = Path(output_dir) / f"batch_{id_pipeline_batch}_{scope}_{DATETIME_SLUG}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Peeking at batch {id_pipeline_batch}, scope: {scope}")
    logger.info(f"Sample type: {sample_type}, n: {n}")
    logger.info(f"Output directory: {output_path}")

    if scope == "detection":
        peek_detect(batch, n, sample_type, output_path)
    if scope == "deduplication":
        peek_dedupe(batch, n, sample_type, output_path)

    logger.info(f"✓ Peek complete! Results saved to: {output_path}")


def peek_detect(batch: PipelineBatch, n: int | str, sample_type: str, output_path: Path):
    """
    Visualize all pipeline results: detections with classifications and captions.
    """
    from models import Caption, Classification
    from textwrap import wrap

    stats = {
        "total_items": 0,
        "total_detections": 0,
        "items_with_detections": 0,
        "failed_captions": 0,
        "failed_classifications": 0,
        "detections_by_scan": 0,
    }

    stats_file = output_path / "stats.txt"
    detections_text_file = output_path / "all_info.txt"

    with open(stats_file, "w") as f:
        f.write(f"Results for Batch {batch.id_pipeline_batch}\n")
        f.write(f"Sample type: {sample_type}\n")
        f.write("=" * 80 + "\n\n")

    with open(detections_text_file, "w") as f:
        f.write(f"All results for Batch {batch.id_pipeline_batch}\n")
        f.write(f"Sample type: {sample_type}\n")
        f.write("=" * 80 + "\n\n")

    if sample_type == "pages":
        _peek_detect_by_pages(batch, n, output_path, stats, stats_file, detections_text_file)
        return

    # Sample volumes (default)
    all_items = batch.items
    if n == "all":
        selected_items = all_items
    else:
        n_samples = min(int(n), len(all_items))
        selected_items = random.sample(all_items, n_samples)

    logger.info(f"Selected {len(selected_items)} volumes out of {len(all_items)}")

    for item in selected_items:
        stats["total_items"] += 1
        volume_barcode = item.ib_volume.barcode

        logger.info(f"Processing volume: {volume_barcode}")

        # Get all detections for this item
        detections = list(
            Detection.select()
            .where(Detection.pipeline_batch_item == item.id_pipeline_batch_item)
            .order_by(Detection.scan_filename, Detection.id_detection)
        )

        if not detections:
            logger.warning(f"No detections found for {volume_barcode}")
            continue

        stats["items_with_detections"] += 1
        stats["total_detections"] += len(detections)

        # Fetch related captions and classifications for all detections
        detection_ids = [det.id_detection for det in detections]

        captions_dict = {
            cap.detection_id: cap
            for cap in Caption.select().where(Caption.detection_id.in_(detection_ids))
        }

        classifications_dict = {
            cls.detection_id: cls
            for cls in Classification.select().where(Classification.detection_id.in_(detection_ids))
        }

        # Attach captions and classifications to detections
        for det in detections:
            det.caption_obj = captions_dict.get(det.id_detection)
            det.classification_obj = classifications_dict.get(det.id_detection)

        # Count failures
        failed_captions = sum(
            1
            for det in detections
            if not det.caption_obj or not det.caption_obj.text or not det.caption_obj.text.strip()
        )
        failed_classifications = sum(
            1
            for det in detections
            if not det.classification_obj or not det.classification_obj.pred_class
        )

        stats["failed_captions"] += failed_captions
        stats["failed_classifications"] += failed_classifications

        # Group detections by scan
        detections_by_scan = {}
        for det in detections:
            if det.scan_filename not in detections_by_scan:
                detections_by_scan[det.scan_filename] = []
            detections_by_scan[det.scan_filename].append(det)

        stats["detections_by_scan"] += len(detections_by_scan)

        # Process ALL scans for the selected volumes
        selected_scans = list(detections_by_scan.keys())

        # Create volume output directory
        volume_output = output_path / volume_barcode
        volume_output.mkdir(exist_ok=True)

        # Load volume data
        try:
            item_data = item.get_data()
        except Exception as e:
            logger.error(f"Could not load data for {volume_barcode}: {e}")
            continue

        # Write volume stats
        with open(stats_file, "a") as f:
            f.write(f"\nVolume: {volume_barcode}\n")
            f.write(f"  Total detections: {len(detections)}\n")
            f.write(f"  Failed captions: {failed_captions}\n")
            f.write(f"  Failed classifications: {failed_classifications}\n")
            f.write(f"  Scans with detections: {len(detections_by_scan)}\n")

        # Write all info to text file
        with open(detections_text_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Volume: {volume_barcode}\n")
            f.write(f"{'='*80}\n\n")
            for scan_filename in sorted(detections_by_scan.keys()):
                f.write(f"\n{scan_filename}:\n{'-'*40}\n")
                for i, det in enumerate(detections_by_scan[scan_filename], 1):
                    f.write(f"\n[{i}] Detection ID: {det.id_detection}\n")

                    # Classification
                    if det.classification_obj and det.classification_obj.pred_class:
                        class_num = str(det.classification_obj.pred_class)
                        class_name = CLASSIFICATION_CLASS_DICT.get(
                            class_num, f"Unknown ({class_num})"
                        )
                        f.write(f"    Class: {class_name} ")
                        f.write(f"(conf: {det.classification_obj.pred_conf:.3f})\n")
                    else:
                        f.write(f"    Class: (FAILED)\n")

                    # Caption
                    caption_text = (
                        det.caption_obj.text
                        if det.caption_obj and det.caption_obj.text
                        else "(FAILED)"
                    )
                    f.write(f"    Caption: {caption_text}\n")

        # Visualize selected scans
        for scan_filename in selected_scans:
            try:
                # Decode full image
                image_bytes = item_data.images.get(scan_filename)
                if image_bytes is None:
                    logger.warning(f"Image {scan_filename} not found in cache")
                    continue

                buffer = np.frombuffer(image_bytes, dtype=np.uint8)
                full_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

                scan_detections = detections_by_scan[scan_filename]

                # Draw numbered bounding boxes on the image
                image_with_boxes = full_image.copy()

                for idx, det in enumerate(scan_detections, 1):
                    bbox = det.bbox_xyxy
                    x1, y1, x2, y2 = map(int, bbox)

                    # Draw rectangle (green if caption exists, red if failed)
                    has_caption = det.caption_obj and det.caption_obj.text
                    color = BBOX_COLOR_CAPTION if has_caption else BBOX_COLOR_NO_CAPTION
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 4)

                    # Draw number in circle at top-left of bbox
                    circle_center = (x1 + 30, y1 + 30)
                    cv2.circle(image_with_boxes, circle_center, 25, color, -1)
                    cv2.putText(
                        image_with_boxes,
                        str(idx),
                        (x1 + 15, y1 + 40),
                        FONT,
                        1.0,
                        BBOX_NUMBER_COLOR,
                        3,
                    )

                # Create legend panel
                font = FONT
                font_scale = LEGEND_FONT_SCALE
                line_spacing = LEGEND_LINE_SPACING
                padding = LEGEND_PADDING
                max_caption_width = MAX_CAPTION_WIDTH

                # Build legend lines
                legend_lines = []
                for idx, det in enumerate(scan_detections, 1):
                    # Detection number and class
                    class_info = ""
                    if det.classification_obj and det.classification_obj.pred_class:
                        det_conf = det.bbox_conf
                        detection_info = f" - Detection Confidence: {det_conf}"
                        class_num = str(det.classification_obj.pred_class)
                        class_name = CLASSIFICATION_CLASS_DICT.get(
                            class_num, f"Unknown ({class_num})"
                        )
                        class_conf = str(det.classification_obj.pred_conf)
                        class_info = f" - {class_name} - Class Confidence: {class_conf}"
                    legend_lines.append(f"[{idx}]{detection_info}{class_info}")

                    # Caption
                    caption_text = (
                        det.caption_obj.text
                        if det.caption_obj and det.caption_obj.text
                        else "(NO CAPTION)"
                    )
                    wrapped = wrap(caption_text, width=max_caption_width)
                    legend_lines.extend([f"  {line}" for line in wrapped])
                    legend_lines.append("")  # Empty line

                legend_height = padding * 2 + len(legend_lines) * line_spacing
                legend_width = image_with_boxes.shape[1]

                # Create white legend panel
                legend_panel = np.full(
                    (legend_height, legend_width, 3), LEGEND_BG_COLOR, dtype=np.uint8
                )

                # Draw legend text
                y_offset = padding + 25
                for line in legend_lines:
                    if line.startswith("["):  # Detection header
                        color = DETECTION_NUMBER_FONT_COLOR
                        weight = DETECTION_NUMBER_WEIGHT
                    else:  # Caption text
                        color = DETECTION_CAPTION_FONT_COLOR
                        weight = DETECTION_CAPTION_WEIGHT

                    cv2.putText(
                        legend_panel,
                        line,
                        (padding, y_offset),
                        font,
                        font_scale,
                        color,
                        weight,
                    )
                    y_offset += line_spacing

                # Create header
                header_height = HEADING_HEIGHT
                header = np.full((header_height, legend_width, 3), HEADING_BG_COLOR, dtype=np.uint8)

                header_text = f"{volume_barcode} | {scan_filename}"
                cv2.putText(
                    header,
                    header_text,
                    (20, 35),
                    FONT,
                    HEADING_FONT_SCALE,
                    HEADING_FONT_COLOR,
                    HEADING_WEIGHT,
                )

                # Classification stats
                cls_success = len(
                    [
                        det
                        for det in scan_detections
                        if det.classification_obj and det.classification_obj.pred_class
                    ]
                )
                cls_failed = len(scan_detections) - cls_success
                cls_text = f"Classifications: {cls_success} | Failed: {cls_failed}"
                cv2.putText(
                    header,
                    cls_text,
                    (20, 65),
                    FONT,
                    SUBHEADING_FONT_SCALE,
                    SUBHEADING_FONT_COLOR,
                    SUBHEADING_WEIGHT,
                )

                # Caption stats
                cap_success = len(
                    [det for det in scan_detections if det.caption_obj and det.caption_obj.text]
                )
                cap_failed = len(scan_detections) - cap_success
                cap_text = f"Captions: {cap_success} | Failed: {cap_failed}"
                cv2.putText(
                    header,
                    cap_text,
                    (20, 90),
                    FONT,
                    SUBHEADING_FONT_SCALE,
                    SUBHEADING_FONT_COLOR,
                    SUBHEADING_WEIGHT,
                )

                # Combine all parts
                output_image = np.vstack([header, image_with_boxes, legend_panel])

                # Save
                output_filename = volume_output / f"{scan_filename.replace('.jp2', '')}_all.jpg"
                cv2.imwrite(
                    str(output_filename), output_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )

                logger.info(f"  Saved: {output_filename.name} ({len(scan_detections)} detections)")

            except Exception as e:
                logger.error(f"Error processing {scan_filename}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Write summary stats
    with open(stats_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total volumes: {stats['total_items']}\n")
        f.write(f"Volumes with detections: {stats['items_with_detections']}\n")
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(f"Failed captions: {stats['failed_captions']}\n")
        f.write(f"Failed classifications: {stats['failed_classifications']}\n")
        f.write(f"Scans processed: {stats['detections_by_scan']}\n")

    logger.info(
        f"Summary: {stats['total_detections']} detections across {stats['detections_by_scan']} scans"
    )


def _peek_detect_by_pages(
    batch: PipelineBatch,
    n: int | str,
    output_path: Path,
    stats: dict,
    stats_file: Path,
    detections_text_file: Path,
):
    """Sample and visualize by random pages (scans with detections)."""
    from models import Caption, Classification
    from textwrap import wrap

    # Create item lookup map
    item_map = {item.id_pipeline_batch_item: item for item in batch.items}
    all_item_ids = list(item_map.keys())

    # Get unique pages efficiently (without loading all detections)
    logger.info("Finding unique pages with detections...")
    unique_pages = list(
        Detection.select(Detection.pipeline_batch_item, Detection.scan_filename)
        .where(Detection.pipeline_batch_item.in_(all_item_ids))
        .distinct()
        .tuples()
    )

    if not unique_pages:
        logger.warning("No detections found in batch")
        return

    all_page_keys = [(item_id, scan) for item_id, scan in unique_pages]
    logger.info(f"Found {len(all_page_keys)} pages with detections")

    # Sample pages
    if n == "all":
        selected_page_keys = all_page_keys
    else:
        n_samples = min(int(n), len(all_page_keys))
        selected_page_keys = random.sample(all_page_keys, n_samples)

    logger.info(f"Selected {len(selected_page_keys)} random pages")

    # Load detections only for selected pages
    logger.info("Loading detections for selected pages...")
    pages = {}
    selected_detections = []
    for item_id, scan_filename in selected_page_keys:
        page_dets = list(
            Detection.select()
            .where(
                (Detection.pipeline_batch_item == item_id) &
                (Detection.scan_filename == scan_filename)
            )
            .order_by(Detection.id_detection)
        )
        pages[(item_id, scan_filename)] = page_dets
        selected_detections.extend(page_dets)

    # Fetch related captions and classifications
    detection_ids = [det.id_detection for det in selected_detections]

    captions_dict = {
        cap.detection_id: cap
        for cap in Caption.select().where(Caption.detection_id.in_(detection_ids))
    }

    classifications_dict = {
        cls.detection_id: cls
        for cls in Classification.select().where(Classification.detection_id.in_(detection_ids))
    }

    # Attach captions and classifications to detections
    for det in selected_detections:
        det.caption_obj = captions_dict.get(det.id_detection)
        det.classification_obj = classifications_dict.get(det.id_detection)

    # Count failures
    failed_captions = sum(
        1
        for det in selected_detections
        if not det.caption_obj or not det.caption_obj.text or not det.caption_obj.text.strip()
    )
    failed_classifications = sum(
        1
        for det in selected_detections
        if not det.classification_obj or not det.classification_obj.pred_class
    )

    stats["total_detections"] = len(selected_detections)
    stats["failed_captions"] = failed_captions
    stats["failed_classifications"] = failed_classifications
    stats["detections_by_scan"] = len(selected_page_keys)

    # Write all info to text file
    with open(detections_text_file, "a") as f:
        f.write(f"\nSampled {len(selected_page_keys)} random pages\n")
        f.write("=" * 80 + "\n\n")

    # Count unique volumes
    unique_items = set(item_id for item_id, _ in selected_page_keys)
    stats["total_items"] = len(unique_items)

    # Create pages output directory
    pages_output = output_path / "pages"
    pages_output.mkdir(exist_ok=True)

    # Identify unique volumes needed
    unique_item_ids = list(set(item_id for item_id, _ in selected_page_keys))
    logger.info(f"Need to load {len(unique_item_ids)} unique volumes...")

    # Load volumes in parallel (I/O-bound - use threads)
    def load_volume(item_id):
        item = item_map.get(item_id)
        if not item:
            return item_id, None, None
        try:
            data = item.get_data()
            return item_id, item.ib_volume.barcode, data
        except Exception as e:
            return item_id, item.ib_volume.barcode, e

    item_data_cache = {}
    num_threads = min(8, len(unique_item_ids))  # Limit concurrent S3 connections

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(load_volume, unique_item_ids))

    for item_id, barcode, result in results:
        if result is None:
            continue
        elif isinstance(result, Exception):
            logger.error(f"Could not load data for {barcode}: {result}")
        else:
            item_data_cache[item_id] = result
            logger.info(f"Loaded volume: {barcode}")

    logger.info(f"Loaded {len(item_data_cache)} volumes")

    # Prepare page data for parallel processing
    logger.info("Preparing page data for processing...")
    page_data_list = []

    for page_idx, (item_id, scan_filename) in enumerate(selected_page_keys, 1):
        item = item_map.get(item_id)
        if not item:
            continue

        volume_barcode = item.ib_volume.barcode
        scan_detections = pages[(item_id, scan_filename)]

        # Get cached volume data
        item_data = item_data_cache.get(item_id)
        if item_data is None:
            continue

        # Write to info file
        with open(detections_text_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Page {page_idx}: {volume_barcode} | {scan_filename}\n")
            f.write(f"{'='*80}\n\n")
            for i, det in enumerate(scan_detections, 1):
                f.write(f"[{i}] Detection ID: {det.id_detection}\n")

                if det.classification_obj and det.classification_obj.pred_class:
                    class_num = str(det.classification_obj.pred_class)
                    class_name = CLASSIFICATION_CLASS_DICT.get(
                        class_num, f"Unknown ({class_num})"
                    )
                    f.write(f"    Class: {class_name} ")
                    f.write(f"(conf: {det.classification_obj.pred_conf:.3f})\n")
                else:
                    f.write(f"    Class: (FAILED)\n")

                caption_text = (
                    det.caption_obj.text
                    if det.caption_obj and det.caption_obj.text
                    else "(FAILED)"
                )
                f.write(f"    Caption: {caption_text}\n\n")

        # Get image bytes
        image_bytes = item_data.images.get(scan_filename)
        if image_bytes is None:
            logger.warning(f"Image {scan_filename} not found in cache")
            continue

        # Convert detections to plain dicts (for pickling to worker processes)
        det_dicts = []
        for det in scan_detections:
            class_name = None
            class_conf = None
            det_conf = det.bbox_conf
            if det.classification_obj and det.classification_obj.pred_class:
                class_num = str(det.classification_obj.pred_class)
                class_name = CLASSIFICATION_CLASS_DICT.get(class_num, f"Unknown ({class_num})")
                class_conf = str(det.classification_obj.pred_conf)

            det_dicts.append({
                "bbox": det.bbox_xyxy,
                "has_caption": bool(det.caption_obj and det.caption_obj.text),
                "caption": det.caption_obj.text if det.caption_obj and det.caption_obj.text else None,
                "class_name": class_name,
                "class_conf": class_conf,
                "det_conf": det_conf,
            })

        page_data_list.append({
            "page_idx": page_idx,
            "volume_barcode": volume_barcode,
            "scan_filename": scan_filename,
            "image_bytes": image_bytes,
            "detections": det_dicts,
            "output_path": str(pages_output),
        })

    # Process pages in parallel
    num_workers = min(multiprocessing.cpu_count(), len(page_data_list))
    logger.info(f"Processing {len(page_data_list)} pages with {num_workers} workers...")

    saved_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_page_image, page_data_list))

    for result in results:
        if result["success"]:
            saved_count += 1
            logger.info(f"  Saved page {result['page_idx']}: {result['filename']} ({result['num_detections']} detections)")
        else:
            logger.error(f"  Failed page {result['page_idx']}: {result.get('error', 'Unknown error')}")

    # Write summary stats
    with open(stats_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY (Page Sampling)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total pages sampled: {stats['detections_by_scan']}\n")
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(f"From volumes: {stats['total_items']}\n")
        f.write(f"Failed captions: {stats['failed_captions']}\n")
        f.write(f"Failed classifications: {stats['failed_classifications']}\n")

    logger.info(f"Summary: Saved {saved_count} pages to {pages_output}")


def peek_dedupe(batch: PipelineBatch, n: int | str, sample_type: str, output_path: Path):
    """
    Visualize deduplication results: groups of similar detections.
    Note: sample_type is accepted but dedupe always samples by volumes.
    """
    from models import DedupedEmbedding, DedupedHash

    _ = sample_type  # Currently only volume sampling supported for dedupe
    logger.info("Fetching deduplication results...")

    # Sample volumes first
    all_items = batch.items
    if n == "all":
        selected_items = all_items
    else:
        n_samples = min(int(n), len(all_items))
        selected_items = random.sample(all_items, n_samples)

    selected_item_ids = [item.id_pipeline_batch_item for item in selected_items]
    logger.info(f"Selected {len(selected_items)} volumes out of {len(all_items)}")

    # Query hash groups for selected volumes only
    hash_records = list(
        DedupedHash.select()
        .where(DedupedHash.pipeline_batch_item.in_(selected_item_ids))
        .order_by(DedupedHash.group_id)
    )

    # Query embedding groups for selected volumes only
    embedding_records = list(
        DedupedEmbedding.select()
        .where(DedupedEmbedding.pipeline_batch_item.in_(selected_item_ids))
        .order_by(DedupedEmbedding.group_id)
    )

    logger.info(f"Found {len(hash_records)} hash dedupe records")
    logger.info(f"Found {len(embedding_records)} embedding dedupe records")

    # Group by group_id
    hash_groups = {}
    for record in hash_records:
        if record.group_id not in hash_groups:
            hash_groups[record.group_id] = []
        hash_groups[record.group_id].append(record)

    embedding_groups = {}
    for record in embedding_records:
        if record.group_id not in embedding_groups:
            embedding_groups[record.group_id] = []
        embedding_groups[record.group_id].append(record)

    # Create output directories
    hash_output = output_path / "hash_groups"
    embedding_output = output_path / "embedding_groups"
    hash_output.mkdir(exist_ok=True)
    embedding_output.mkdir(exist_ok=True)

    # Process hash groups (show ALL groups from selected volumes)
    logger.info(f"Processing {len(hash_groups)} hash groups...")
    _process_dedupe_groups(hash_groups, hash_output, "hash", batch)

    # Process embedding groups (show ALL groups from selected volumes)
    logger.info(f"Processing {len(embedding_groups)} embedding groups...")
    _process_dedupe_groups(embedding_groups, embedding_output, "embedding", batch)

    # Write summary stats
    stats_file = output_path / "dedupe_stats.txt"
    with open(stats_file, "w") as f:
        f.write(f"Deduplication Results for Batch {batch.id_pipeline_batch}\n")
        f.write(f"(Showing results for {len(selected_items)} selected volumes)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Hash Groups: {len(hash_groups)}\n")
        f.write(f"  Total detections: {len(hash_records)}\n")
        f.write(
            f"  Avg group size: {len(hash_records) / len(hash_groups):.1f}\n"
            if hash_groups
            else "  Avg group size: 0\n"
        )
        f.write(
            f"  Largest group: {max(len(g) for g in hash_groups.values())}\n"
            if hash_groups
            else "  Largest group: 0\n"
        )
        f.write(f"\nEmbedding Groups: {len(embedding_groups)}\n")
        f.write(f"  Total detections: {len(embedding_records)}\n")
        f.write(
            f"  Avg group size: {len(embedding_records) / len(embedding_groups):.1f}\n"
            if embedding_groups
            else "  Avg group size: 0\n"
        )
        f.write(
            f"  Largest group: {max(len(g) for g in embedding_groups.values())}\n"
            if embedding_groups
            else "  Largest group: 0\n"
        )

        # Write group size distribution
        f.write("\n" + "=" * 80 + "\n")
        f.write("Hash Group Size Distribution:\n")
        f.write("-" * 40 + "\n")
        hash_size_dist = {}
        for group in hash_groups.values():
            size = len(group)
            hash_size_dist[size] = hash_size_dist.get(size, 0) + 1
        for size in sorted(hash_size_dist.keys()):
            f.write(f"  Size {size}: {hash_size_dist[size]} groups\n")

        f.write("\nEmbedding Group Size Distribution:\n")
        f.write("-" * 40 + "\n")
        emb_size_dist = {}
        for group in embedding_groups.values():
            size = len(group)
            emb_size_dist[size] = emb_size_dist.get(size, 0) + 1
        for size in sorted(emb_size_dist.keys()):
            f.write(f"  Size {size}: {emb_size_dist[size]} groups\n")

    logger.info(f"Dedupe stats written to: {stats_file}")


def _process_dedupe_groups(
    groups: dict,
    output_path: Path,
    group_type: str,
    batch: PipelineBatch,
):
    """
    Helper to process and visualize dedupe groups.
    """
    # Process ALL groups (no sampling)
    group_ids = list(groups.keys())
    selected_group_ids = sorted(group_ids, key=lambda gid: len(groups[gid]), reverse=True)

    # Create a mapping of pipeline_batch_item_id to item for fast lookup
    item_map = {item.id_pipeline_batch_item: item for item in batch.items}

    for group_id in selected_group_ids:
        records = groups[group_id]
        group_folder = output_path / f"group_{group_id}_size_{len(records)}"
        group_folder.mkdir(exist_ok=True)

        logger.info(f"  Processing {group_type} group {group_id} ({len(records)} detections)...")

        # Write group info
        info_file = group_folder / "group_info.txt"
        with open(info_file, "w") as f:
            f.write(f"Group ID: {group_id}\n")
            f.write(f"Type: {group_type}\n")
            f.write(f"Size: {len(records)}\n")
            f.write("=" * 80 + "\n\n")

            for idx, record in enumerate(records, 1):
                f.write(f"[{idx}] Detection ID: {record.detection.id_detection}\n")
                f.write(f"    Scan: {record.scan_filename}\n")
                item = item_map.get(record.pipeline_batch_item.id_pipeline_batch_item)
                if item:
                    f.write(f"    Volume: {item.ib_volume.barcode}\n")
                if group_type == "hash":
                    f.write(f"    Hash: {record.image_hash}\n")
                f.write("\n")

        # Extract and save detection images
        for idx, record in enumerate(records, 1):
            try:
                # Get the item
                item = item_map.get(record.pipeline_batch_item.id_pipeline_batch_item)
                if not item:
                    logger.warning(f"Item not found for detection {record.detection.id_detection}")
                    continue

                # Load item data
                item_data = item.get_data()

                # Get the image
                image_bytes = item_data.images.get(record.scan_filename)
                if image_bytes is None:
                    logger.warning(f"Image {record.scan_filename} not found")
                    continue

                # Decode image
                buffer = np.frombuffer(image_bytes, dtype=np.uint8)
                full_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

                # Crop detection
                detection = record.detection
                bbox = detection.bbox_xyxy
                x1, y1, x2, y2 = map(int, bbox)

                # Ensure bounds are within image
                h, w = full_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cropped = full_image[y1:y2, x1:x2]

                # Create filename
                volume_barcode = item.ib_volume.barcode if item else "unknown"
                scan_name = record.scan_filename.replace(".jp2", "").replace(".tiff", "")
                filename = f"{idx:02d}_{volume_barcode}_{scan_name}_det{detection.id_detection}.jpg"

                # Save
                output_file = group_folder / filename
                cv2.imwrite(str(output_file), cropped, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            except Exception as e:
                logger.error(f"Error processing detection {record.detection.id_detection}: {e}")
                continue

        logger.info(f"    Saved {len(records)} images to {group_folder.name}")
