import click
from loguru import logger
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import random

from models import PipelineBatch, Detection
from utils import get_db
from const import CLASS_DICT


@click.command("peek")
@click.option("--step", type=click.Choice(["detect", "dedupe"]), required=True)
@click.option("--id-pipeline-batch", type=int, required=True, help="Pipeline batch ID to inspect")
@click.option("--n", default=5, help="Number of random volumes to select (integer or 'all')")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./peek_output",
    help="Output directory for visualization",
)
def peek(step, id_pipeline_batch, n, output_dir):
    """
    Visualize pipeline outputs for debugging and validation.

    Examples:
        peek --step detect --id-pipeline-batch 123 --n 5
        peek --step detect --id-pipeline-batch 123 --n all
    """
    get_db()

    # Validate pipeline batch exists
    try:
        batch = PipelineBatch.get_by_id(id_pipeline_batch)
    except Exception as e:
        logger.error(f"Pipeline batch {id_pipeline_batch} not found")
        return

    output_path = (
        Path(output_dir)
        / f"batch_{id_pipeline_batch}_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Peeking at batch {id_pipeline_batch}, step: {step}")
    logger.info(f"Output directory: {output_path}")

    if step == "detect":
        peek_detect(batch, n, output_path)
    if step == "dedupe":
        peek_dedupe(batch, n, output_path)

    logger.info(f"✓ Peek complete! Results saved to: {output_path}")


def peek_detect(batch: PipelineBatch, n: int | str, output_path: Path):
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
        f.write("=" * 80 + "\n\n")

    with open(detections_text_file, "w") as f:
        f.write(f"All results for Batch {batch.id_pipeline_batch}\n")
        f.write("=" * 80 + "\n\n")

    # Sample volumes
    all_items = batch.items[:100]
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
            if not det.caption_obj
            or not det.caption_obj.caption
            or not det.caption_obj.caption.strip()
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
                        class_name = CLASS_DICT.get(class_num, f"Unknown ({class_num})")
                        f.write(f"    Class: {class_name} ")
                        f.write(f"(conf: {det.classification_obj.pred_conf:.3f})\n")
                    else:
                        f.write(f"    Class: (FAILED)\n")

                    # Caption
                    caption_text = (
                        det.caption_obj.caption
                        if det.caption_obj and det.caption_obj.caption
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
                    has_caption = det.caption_obj and det.caption_obj.caption
                    color = (0, 255, 0) if has_caption else (0, 0, 255)
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 4)

                    # Draw number in circle at top-left of bbox
                    circle_center = (x1 + 30, y1 + 30)
                    cv2.circle(image_with_boxes, circle_center, 25, color, -1)
                    cv2.putText(
                        image_with_boxes,
                        str(idx),
                        (x1 + 15, y1 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        3,
                    )

                # Create legend panel
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                line_spacing = 35
                padding = 20
                max_caption_width = 100

                # Build legend lines
                legend_lines = []
                for idx, det in enumerate(scan_detections, 1):
                    # Detection number and class
                    class_info = ""
                    if det.classification_obj and det.classification_obj.pred_class:
                        class_num = str(det.classification_obj.pred_class)
                        class_name = CLASS_DICT.get(class_num, f"Unknown ({class_num})")
                        class_info = f" - {class_name}"
                    legend_lines.append(f"[{idx}]{class_info}")

                    # Caption
                    caption_text = (
                        det.caption_obj.caption
                        if det.caption_obj and det.caption_obj.caption
                        else "(NO CAPTION)"
                    )
                    wrapped = wrap(caption_text, width=max_caption_width)
                    legend_lines.extend([f"  {line}" for line in wrapped])
                    legend_lines.append("")  # Empty line

                legend_height = padding * 2 + len(legend_lines) * line_spacing
                legend_width = image_with_boxes.shape[1]

                # Create white legend panel
                legend_panel = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

                # Draw legend text
                y_offset = padding + 25
                for line in legend_lines:
                    if line.startswith("["):  # Detection header
                        color = (0, 0, 0)
                        weight = 3
                    else:  # Caption text
                        color = (50, 50, 50)
                        weight = 1

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
                header_height = 100
                header = np.zeros((header_height, legend_width, 3), dtype=np.uint8)

                header_text = f"{volume_barcode} | {scan_filename}"
                cv2.putText(
                    header, header_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
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
                    header, cls_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
                )

                # Caption stats
                cap_success = len(
                    [det for det in scan_detections if det.caption_obj and det.caption_obj.caption]
                )
                cap_failed = len(scan_detections) - cap_success
                cap_text = f"Captions: {cap_success} | Failed: {cap_failed}"
                cv2.putText(
                    header, cap_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
                )

                # Combine all parts
                output_image = np.vstack([header, image_with_boxes, legend_panel])

                # Save
                output_filename = volume_output / f"{scan_filename.replace('.jp2', '')}_all.jpg"
                cv2.imwrite(str(output_filename), output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

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


def peek_dedupe(batch: PipelineBatch, n: int | str, output_path: Path):
    """
    Visualize deduplication results: groups of similar detections.
    """
    from models import DedupedEmbedding, DedupedHash

    logger.info("Fetching deduplication results...")

    # Sample volumes first
    all_items = batch.items[:100]
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
                scan_name = record.scan_filename.replace(".jp2", "").replace(".jpg", "")
                filename = f"{idx:02d}_{volume_barcode}_{scan_name}_det{detection.id_detection}.jpg"

                # Save
                output_file = group_folder / filename
                cv2.imwrite(str(output_file), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])

            except Exception as e:
                logger.error(f"Error processing detection {record.detection.id_detection}: {e}")
                continue

        logger.info(f"    Saved {len(records)} images to {group_folder.name}")
