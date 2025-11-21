# NOTE: This command's goal is to help us have a look at some sample data to make sure the pipeline behaves as expected

import click
from loguru import logger
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import random

from models import PipelineBatch, PipelineBatchItem, Detection, Classification
from utils import get_db


@click.command("peek")
@click.option(
    "--step", type=click.Choice(["detect", "classify", "dedupe", "captions"]), required=True
)
@click.option("--id-pipeline-batch", type=int, required=True, help="Pipeline batch ID to inspect")
@click.option("--n", default=5, help="Number of samples to show per volume (integer or 'all')")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./peek_output",
    help="Output directory for visualization",
)
@click.option(
    "--random-sample", is_flag=True, help="Randomly sample images instead of taking first N"
)
def peek(step, id_pipeline_batch, n, output_dir, random_sample):
    """
    Visualize pipeline outputs for debugging and validation.

    Examples:
        peek --step detect --id-pipeline-batch 123 --n 5
        peek --step detect --id-pipeline-batch 123 --n all --random-sample
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
        peek_detect(batch, n, output_path, random_sample)
    elif step == "classify":
        peek_cls(batch, n, output_path, random_sample)
    elif step == "dedupe":
        peek_dedupe(batch, n, output_path, random_sample)
    elif step == "captions":
        peek_captions(batch, n, output_path, random_sample)

    logger.info(f"✓ Peek complete! Results saved to: {output_path}")


def peek_detect(batch: PipelineBatch, n: int | str, output_path: Path, random_sample: bool):
    """
    Visualize detection results: bounding boxes drawn on images.
    """
    stats = {
        "total_items": 0,
        "total_scans": 0,
        "total_detections": 0,
        "items_with_detections": 0,
        "scans_with_detections": 0,
    }

    stats_file = output_path / "detection_stats.txt"

    with open(stats_file, "w") as f:
        f.write(f"Detection Results for Batch {batch.id_pipeline_batch}\n")
        f.write("=" * 80 + "\n\n")

    for item in batch.items:
        stats["total_items"] += 1
        volume_barcode = item.ib_volume.barcode

        logger.info(f"Processing volume: {volume_barcode}")

        # Get all detections for this item
        detections = list(
            Detection.select()
            .where(Detection.pipeline_batch_item == item.id_pipeline_batch_item)
            .order_by(Detection.scan_filename)
        )

        if not detections:
            logger.warning(f"No detections found for {volume_barcode}")
            continue

        stats["items_with_detections"] += 1
        stats["total_detections"] += len(detections)

        # Group detections by scan
        detections_by_scan = {}
        for det in detections:
            if det.scan_filename not in detections_by_scan:
                detections_by_scan[det.scan_filename] = []
            detections_by_scan[det.scan_filename].append(det)

        stats["total_scans"] += len(detections_by_scan)
        stats["scans_with_detections"] += len(detections_by_scan)

        # Sample scans to visualize
        scan_filenames = list(detections_by_scan.keys())

        if n == "all":
            selected_scans = scan_filenames
        else:
            n_samples = min(int(n), len(scan_filenames))
            if random_sample:
                selected_scans = random.sample(scan_filenames, n_samples)
            else:
                selected_scans = scan_filenames[:n_samples]

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
            f.write(f"  Total scans: {len(item_data.images)}\n")
            f.write(f"  Scans with detections: {len(detections_by_scan)}\n")
            f.write(f"  Total detections: {len(detections)}\n")
            f.write(f"  Avg detections per scan: {len(detections)/len(detections_by_scan):.2f}\n")

        # Visualize selected scans
        for scan_filename in selected_scans:
            try:
                # Decode image
                image_bytes = item_data.images.get(scan_filename)
                if image_bytes is None:
                    logger.warning(f"Image {scan_filename} not found in cache")
                    continue

                buffer = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

                # Draw bounding boxes
                scan_detections = detections_by_scan[scan_filename]

                for det in scan_detections:
                    bbox = det.bbox_xyxy
                    x1, y1, x2, y2 = map(int, bbox)
                    conf = det.bbox_conf

                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Add confidence label
                    label = f"{conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(
                        image,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
                    )

                # Add header with info
                header_height = 60
                header = np.zeros((header_height, image.shape[1], 3), dtype=np.uint8)
                header_text = (
                    f"{volume_barcode} | {scan_filename} | Detections: {len(scan_detections)}"
                )
                cv2.putText(
                    header, header_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )

                # Combine header and image
                output_image = np.vstack([header, image])

                # Save
                output_filename = (
                    volume_output / f"{scan_filename.replace('.jp2', '')}_detections.jpg"
                )
                cv2.imwrite(str(output_filename), output_image)

                logger.info(f"  Saved: {output_filename.name} ({len(scan_detections)} detections)")

            except Exception as e:
                logger.error(f"Error processing {scan_filename}: {e}")
                continue

    # Write summary stats
    with open(stats_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total volumes: {stats['total_items']}\n")
        f.write(f"Volumes with detections: {stats['items_with_detections']}\n")
        f.write(f"Total scans processed: {stats['total_scans']}\n")
        f.write(f"Scans with detections: {stats['scans_with_detections']}\n")
        f.write(f"Total detections: {stats['total_detections']}\n")
        if stats["scans_with_detections"] > 0:
            f.write(
                f"Avg detections per scan: {stats['total_detections']/stats['scans_with_detections']:.2f}\n"
            )

    logger.info(
        f"Summary: {stats['total_detections']} detections across {stats['scans_with_detections']} scans"
    )


def peek_cls(batch: PipelineBatch, n: int | str, output_path: Path, random_sample: bool):
    """
    Visualize classification results: crops with their predicted classes.
    """
    logger.info("Classification peek not yet implemented")
    # TODO: Show cropped regions with their classification labels
    pass


def peek_dedupe(batch: PipelineBatch, n: int | str, output_path: Path, random_sample: bool):
    """
    Visualize deduplication results: groups of similar detections.
    """
    logger.info("Deduplication peek not yet implemented")
    # TODO: Show groups of duplicates side by side
    pass


def peek_captions(batch: PipelineBatch, n: int | str, output_path: Path, random_sample: bool):
    """
    Visualize captioning results: cropped regions with their generated captions.
    """
    from models import Caption
    from textwrap import wrap

    stats = {
        "total_items": 0,
        "total_captions": 0,
        "items_with_captions": 0,
        "failed_captions": 0,
        "captions_by_scan": 0,
    }

    stats_file = output_path / "caption_stats.txt"
    captions_text_file = output_path / "all_captions.txt"

    with open(stats_file, "w") as f:
        f.write(f"Caption Results for Batch {batch.id_pipeline_batch}\n")
        f.write("=" * 80 + "\n\n")

    with open(captions_text_file, "w") as f:
        f.write(f"All Captions for Batch {batch.id_pipeline_batch}\n")
        f.write("=" * 80 + "\n\n")

    for item in batch.items[:100]:
        stats["total_items"] += 1
        volume_barcode = item.ib_volume.barcode

        logger.info(f"Processing volume: {volume_barcode}")

        # Get all captions for this item
        captions = list(
            Caption.select(Caption, Detection)
            .join(Detection, on=(Caption.detection_id == Detection.id_detection))
            .where(Caption.pipeline_batch_item == item.id_pipeline_batch_item)
            .order_by(Caption.scan_filename, Detection.id_detection)
        )

        if not captions:
            logger.warning(f"No captions found for {volume_barcode}")
            continue

        stats["items_with_captions"] += 1

        # Count failed captions (empty or None)
        failed = sum(1 for c in captions if not c.caption or c.caption.strip() == "")
        stats["failed_captions"] += failed
        valid_captions = [c for c in captions if c.caption and c.caption.strip()]
        stats["total_captions"] += len(valid_captions)

        # Group captions by scan
        captions_by_scan = {}
        for cap in captions:
            if cap.scan_filename not in captions_by_scan:
                captions_by_scan[cap.scan_filename] = []
            captions_by_scan[cap.scan_filename].append(cap)

        stats["captions_by_scan"] += len(captions_by_scan)

        # Sample scans to visualize
        scan_filenames = list(captions_by_scan.keys())

        if n == "all":
            selected_scans = scan_filenames
        else:
            n_samples = min(int(n), len(scan_filenames))
            if random_sample:
                selected_scans = random.sample(scan_filenames, n_samples)
            else:
                selected_scans = scan_filenames[:n_samples]

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
            f.write(f"  Total captions: {len(captions)}\n")
            f.write(f"  Valid captions: {len(valid_captions)}\n")
            f.write(f"  Failed captions: {failed}\n")
            f.write(f"  Scans with captions: {len(captions_by_scan)}\n")
            if valid_captions:
                avg_len = sum(len(c.caption) for c in valid_captions) / len(valid_captions)
                f.write(f"  Avg caption length: {avg_len:.1f} chars\n")

        # Write all captions to text file
        with open(captions_text_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Volume: {volume_barcode}\n")
            f.write(f"{'='*80}\n\n")
            for scan_filename in sorted(captions_by_scan.keys()):
                f.write(f"\n{scan_filename}:\n{'-'*40}\n")
                for i, cap in enumerate(captions_by_scan[scan_filename], 1):
                    f.write(f"\n[{i}] Detection ID: {cap.detection_id}\n")
                    f.write(f"    Caption: {cap.caption or '(FAILED)'}\n")

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

                scan_captions = captions_by_scan[scan_filename]

                # Draw numbered bounding boxes on the image
                image_with_boxes = full_image.copy()

                for idx, cap in enumerate(scan_captions, 1):
                    detection = cap.detection
                    bbox = detection.bbox_xyxy
                    x1, y1, x2, y2 = map(int, bbox)

                    # Draw rectangle
                    color = (
                        (0, 255, 0) if cap.caption else (0, 0, 255)
                    )  # Green if caption exists, red if failed
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

                # Create caption legend panel
                # Calculate required height
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                line_spacing = 40
                padding = 20
                max_caption_width = 120  # characters per line

                # Calculate total height needed
                legend_lines = []
                for idx, cap in enumerate(scan_captions, 1):
                    caption_text = cap.caption or "(CAPTION FAILED)"
                    wrapped = wrap(caption_text, width=max_caption_width)
                    legend_lines.append(f"[{idx}]")
                    legend_lines.extend(wrapped)
                    legend_lines.append("")  # Empty line between captions

                legend_height = padding * 2 + len(legend_lines) * line_spacing
                legend_width = image_with_boxes.shape[1]

                # Create white legend panel
                legend_panel = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

                # Draw captions
                y_offset = padding + 30
                for line in legend_lines:
                    if line.startswith("["):  # Caption number
                        color = (0, 0, 0)
                        weight = 3
                    else:  # Caption text
                        color = (50, 50, 50)
                        weight = 2

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
                header_height = 80
                header = np.zeros((header_height, legend_width, 3), dtype=np.uint8)
                header_text = f"{volume_barcode} | {scan_filename}"
                cv2.putText(
                    header, header_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
                )
                stats_text = f"Captions: {len([c for c in scan_captions if c.caption])} | Failed: {len([c for c in scan_captions if not c.caption])}"
                cv2.putText(
                    header, stats_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
                )

                # Combine all parts
                output_image = np.vstack([header, image_with_boxes, legend_panel])

                # Save
                output_filename = (
                    volume_output / f"{scan_filename.replace('.jp2', '')}_captions.jpg"
                )
                cv2.imwrite(str(output_filename), output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

                logger.info(f"  Saved: {output_filename.name} ({len(scan_captions)} captions)")

            except Exception as e:
                logger.error(f"Error processing {scan_filename}: {e}")
                import traceback

                traceback.print_exc()
                continue

    # Calculate average caption length
    if stats["total_captions"] > 0:
        total_length = 0
        count = 0
        for item in batch.items:
            for cap in Caption.select().where(
                (Caption.pipeline_batch_item == item.id_pipeline_batch_item)
                & (Caption.caption.is_null(False))
            ):
                if cap.caption:
                    total_length += len(cap.caption)
                    count += 1

        if count > 0:
            stats["avg_caption_length"] = total_length / count

    # Write summary stats
    with open(stats_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total volumes: {stats['total_items']}\n")
        f.write(f"Volumes with captions: {stats['items_with_captions']}\n")
        f.write(f"Total captions: {stats['total_captions']}\n")
        f.write(f"Failed captions: {stats['failed_captions']}\n")
        f.write(
            f"Success rate: {(stats['total_captions']/(stats['total_captions']+stats['failed_captions'])*100):.1f}%\n"
            if (stats["total_captions"] + stats["failed_captions"]) > 0
            else "Success rate: N/A\n"
        )
        f.write(f"Scans with captions: {stats['captions_by_scan']}\n")
        if stats.get("avg_caption_length"):
            f.write(f"Avg caption length: {stats['avg_caption_length']:.1f} characters\n")

    logger.info(
        f"Summary: {stats['total_captions']} captions across {stats['captions_by_scan']} scans "
        f"({stats['failed_captions']} failed)"
    )
