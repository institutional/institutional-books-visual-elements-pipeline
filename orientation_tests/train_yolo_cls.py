"""
Train a YOLO26m-cls orientation classifier.

Reads manual_labels.json, organizes images into class folders, and fine-tunes
YOLO26m-cls for 4-class orientation prediction. Predictions represent the
correction needed to make an image upright.

Data is split into train/val/test (80/10/10). The test set is never used during
training and is used for evaluation PDF generation.

Usage:
    python orientation_tests/train_yolo_cls.py [--labels orientation_tests/manual_labels.json] [--epochs 50]
    python orientation_tests/train_yolo_cls.py --no-synthetic --eval-pdf orientation_tests/eval_yolo.pdf
"""

import argparse
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import download_bucket_files
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm
from ultralytics import YOLO

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
BUCKET_REPO = "institutional/institutional-books-hl-visual-elements-images"

CLASS_NAMES = ["upright", "rotated_90_clockwise", "rotated_180", "rotated_90_counterclockwise"]

LABEL_TO_QUARTERS = {
    "upright": 0,
    "rotated_90_clockwise": 1,
    "rotated_180": 2,
    "rotated_90_counterclockwise": 3,
}

DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 3


def load_image_safely(path: str) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGB", "L"):
            return img.convert("RGB")
        rgba_img = img.convert("RGBA")
        background = Image.new("RGB", rgba_img.size, (255, 255, 255))
        background.paste(rgba_img, mask=rgba_img)
        return background


def apply_rotation(img: Image.Image, quarters_cw: int) -> Image.Image:
    degrees_map = {0: 0, 1: 270, 2: 180, 3: 90}
    q = quarters_cw % 4
    if q == 0:
        return img
    return img.rotate(degrees_map[q], expand=True)


def download_images(filenames: list[str], image_dir: str):
    logger.info(f"Downloading {len(filenames)} images...")
    file_pairs = [(fn, os.path.join(image_dir, fn)) for fn in filenames]
    batch_size = 200
    for i in tqdm(range(0, len(file_pairs), batch_size), desc="Downloading"):
        batch = file_pairs[i : i + batch_size]
        for attempt in range(DOWNLOAD_MAX_RETRIES):
            try:
                start = time.time()
                download_bucket_files(BUCKET_REPO, files=batch, token=HF_TOKEN)
                elapsed = time.time() - start
                if elapsed > DOWNLOAD_TIMEOUT:
                    logger.warning(f"Batch took {elapsed:.0f}s")
                break
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1}/{DOWNLOAD_MAX_RETRIES} failed: {e}")
                time.sleep(2 ** attempt)
                if attempt == DOWNLOAD_MAX_RETRIES - 1:
                    logger.error(f"Skipping batch at {i}")


def build_dataset(labels: list, image_dir: str, dataset_dir: str, split_name: str, no_synthetic: bool = False):
    """Write images into class folders for a single split (train or val).

    Class folders represent the correction needed.
    """
    for cls_name in CLASS_NAMES:
        os.makedirs(os.path.join(dataset_dir, split_name, cls_name), exist_ok=True)

    skipped = 0
    for entry in tqdm(labels, desc=f"Building {split_name}"):
        filename = entry["filename"]
        src_path = os.path.join(image_dir, filename)
        if not os.path.exists(src_path):
            skipped += 1
            continue

        try:
            img = load_image_safely(src_path)
        except Exception:
            skipped += 1
            continue

        stem = Path(filename).stem
        correction = entry["correction_quarters"]

        if no_synthetic:
            cls_name = CLASS_NAMES[correction]
            out_path = os.path.join(dataset_dir, split_name, cls_name, f"{stem}.jpg")
            img.save(out_path, "JPEG", quality=90)
        else:
            if correction:
                img = apply_rotation(img, correction)
            # img is now upright; create 4 rotations
            for rot_class in range(4):
                rotated = apply_rotation(img, rot_class) if rot_class else img
                # Label = correction needed = inverse of applied rotation
                correction_label = (4 - rot_class) % 4
                cls_name = CLASS_NAMES[correction_label]
                out_path = os.path.join(dataset_dir, split_name, cls_name, f"{stem}_r{rot_class}.jpg")
                rotated.save(out_path, "JPEG", quality=90)

    if skipped:
        logger.warning(f"{split_name}: skipped {skipped} images (missing or corrupt)")

    for cls_name in CLASS_NAMES:
        count = len(os.listdir(os.path.join(dataset_dir, split_name, cls_name)))
        logger.info(f"  {split_name}/{cls_name}: {count}")


# --- Eval PDF generation ---

def image_to_bytesio(img: Image.Image, max_dim: int = 300) -> io.BytesIO:
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_eval_pdf(yolo_model, test_labels: list, image_dir: str, output_path: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepTogether,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet

    results = []
    for entry in tqdm(test_labels, desc="Running eval on test set"):
        path = os.path.join(image_dir, entry["filename"])
        if not os.path.exists(path):
            continue
        try:
            img = load_image_safely(path)
            yolo_results = yolo_model(img, verbose=False)
            probs = yolo_results[0].probs
            pred_name = yolo_results[0].names[probs.top1]
            pred = LABEL_TO_QUARTERS.get(pred_name, 0)
            confidence = float(probs.top1conf)
            if pred != 0:
                results.append({
                    "filename": entry["filename"],
                    "prediction": pred,
                    "confidence": confidence,
                    "image": img,
                })
        except Exception as e:
            logger.warning(f"Eval failed on {entry['filename']}: {e}")

    logger.info(f"Eval: {len(results)} corrections out of {len(test_labels)} test images")

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.5 * inch, rightMargin=0.5 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("YOLO Orientation Predictions (Test Set)", styles["Title"]))
    elements.append(Paragraph(
        f"{len(results)} corrected images from {len(test_labels)} test samples",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.3 * inch))

    img_width = 2.8 * inch
    img_height = 2.8 * inch

    for entry in results:
        try:
            original = entry["image"]
            pred_class = entry["prediction"]
            label = CLASS_NAMES[pred_class]
            confidence = entry["confidence"]
            corrected = apply_rotation(original, pred_class)

            orig_buf = image_to_bytesio(original.copy())
            cor_buf = image_to_bytesio(corrected.copy())

            orig_img = RLImage(orig_buf, width=img_width, height=img_height, kind="proportional")
            cor_img = RLImage(cor_buf, width=img_width, height=img_height, kind="proportional")

            header_table = Table(
                [[Paragraph("<b>Original</b>", styles["Normal"]),
                  Paragraph("<b>Corrected</b>", styles["Normal"])]],
                colWidths=[3.5 * inch, 3.5 * inch],
            )
            img_table = Table(
                [[orig_img, cor_img]],
                colWidths=[3.5 * inch, 3.5 * inch],
                rowHeights=[img_height + 0.1 * inch],
            )
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))

            caption = Paragraph(
                f"<b>{entry['filename']}</b> — correction: {label} (confidence: {confidence:.1%})",
                styles["Normal"],
            )
            block = KeepTogether([
                caption, Spacer(1, 0.1 * inch), header_table, img_table, Spacer(1, 0.3 * inch),
            ])
            elements.append(block)
        except Exception as e:
            logger.warning(f"PDF: failed to add {entry.get('filename', '?')}: {e}")

    doc.build(elements)
    logger.info(f"Eval PDF saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26m-cls orientation model")
    parser.add_argument("--labels", type=str, default="orientation_tests/manual_labels.json")
    parser.add_argument("--output-dir", type=str, default="orientation_tests/model")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Reuse existing image directory (skip download)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Train on raw images with their labels as-is, no correction + 4-rotation augmentation")
    parser.add_argument("--eval-pdf", type=str, default=None,
                        help="After training, generate an evaluation PDF at this path")
    args = parser.parse_args()

    with open(args.labels) as f:
        data = json.load(f)

    results = data["results"][:args.max_samples] if args.max_samples else data["results"]
    labels = []
    for r in results:
        if "error" in r:
            continue
        label_field = r.get("manual_label")
        if not label_field or label_field not in LABEL_TO_QUARTERS:
            continue
        labels.append({
            "filename": r["filename"],
            "correction_quarters": LABEL_TO_QUARTERS[label_field],
        })
    logger.info(f"Using {len(labels)} labels")

    if not labels:
        logger.error("No valid labels found.")
        return

    # Split into train/val/test
    random.shuffle(labels)
    train_size = int(len(labels) * args.train_split)
    val_size = int(len(labels) * args.val_split)
    train_labels = labels[:train_size]
    val_labels = labels[train_size:train_size + val_size]
    test_labels = labels[train_size + val_size:]
    logger.info(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    # Get images
    if args.image_dir:
        image_dir = args.image_dir
        logger.info(f"Using existing image dir: {image_dir}")
    else:
        image_dir = os.path.join(args.output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        existing = set(os.listdir(image_dir))
        to_download = [l["filename"] for l in labels if l["filename"] not in existing]
        if to_download:
            download_images(to_download, image_dir)
        else:
            logger.info("All images already downloaded")

    # Build dataset structure (train + val only; test is held out)
    dataset_dir = os.path.join(args.output_dir, "dataset")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    logger.info("Building classification dataset...")
    build_dataset(train_labels, image_dir, dataset_dir, "train", no_synthetic=args.no_synthetic)
    build_dataset(val_labels, image_dir, dataset_dir, "val", no_synthetic=args.no_synthetic)

    # Train
    model = YOLO("yolo26m-cls.pt")
    train_results = model.train(
        data=dataset_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=args.output_dir,
        name="yolo26m_orientation",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        workers=4,
    )

    best_model_path = os.path.join(args.output_dir, "yolo26m_orientation", "weights", "best.pt")
    logger.info(f"Training complete. Best model: {best_model_path}")

    # Generate eval PDF if requested
    if args.eval_pdf and os.path.exists(best_model_path):
        eval_model = YOLO(best_model_path)
        generate_eval_pdf(eval_model, test_labels, image_dir, args.eval_pdf)


if __name__ == "__main__":
    main()
