"""
Train a YOLO26m-cls orientation classifier.

Reads manual_labels.json, organizes images into class folders, and fine-tunes
YOLO26m-cls for 4-class orientation prediction. Predictions represent the
correction needed to make an image upright.

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
import tempfile
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import download_bucket_files
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm
from ultralytics import YOLO

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
BUCKET_REPO = "institutional/institutional-books-hl-visual-elements-images"
DATASET_REPO = "institutional/institutional-books-hl-visual-elements"
ALLOWED_SPLITS = ["image_illustration", "chart_graph"]

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


def build_dataset(labels: list, image_dir: str, dataset_dir: str, val_split: float = 0.15, no_synthetic: bool = False):
    """Organize images into train/val class folders.

    Class folders represent the correction needed. With synthetic augmentation,
    each upright image is rotated 4 ways and placed in the folder for its correction.
    """
    random.shuffle(labels)
    val_size = int(len(labels) * val_split)
    splits = {"val": labels[:val_size], "train": labels[val_size:]}

    for split_name, split_labels in splits.items():
        for cls_name in CLASS_NAMES:
            os.makedirs(os.path.join(dataset_dir, split_name, cls_name), exist_ok=True)

        skipped = 0
        for entry in tqdm(split_labels, desc=f"Building {split_name}"):
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

    for split_name in splits:
        for cls_name in CLASS_NAMES:
            count = len(os.listdir(os.path.join(dataset_dir, split_name, cls_name)))
            logger.info(f"  {split_name}/{cls_name}: {count}")


# --- Eval PDF generation ---

def sample_eval_filenames(sample_size: int, exclude: set) -> list[str]:
    logger.info(f"Sampling {sample_size} eval filenames (excluding {len(exclude)} known)...")
    per_split = sample_size // len(ALLOWED_SPLITS)
    filenames = []

    for split in ALLOWED_SPLITS:
        ds = load_dataset(DATASET_REPO, split=split, streaming=True, token=HF_TOKEN)
        reservoir = []
        seen = 0
        for row in ds:
            stem = Path(row["page_filename_src"]).stem
            fn = f"{row['barcode_src']}_{stem}_{row['id']}.webp"
            if fn in exclude:
                continue
            seen += 1
            if seen <= per_split:
                reservoir.append(fn)
            else:
                j = random.randint(0, seen - 1)
                if j < per_split:
                    reservoir[j] = fn
            if seen >= per_split * 10:
                break
        filenames.extend(reservoir)

    random.shuffle(filenames)
    return filenames[:sample_size]


def image_to_bytesio(img: Image.Image, max_dim: int = 300) -> io.BytesIO:
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def generate_eval_pdf(yolo_model, labels_file: str, output_path: str, num_samples: int = 500):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepTogether,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet

    exclude = set()
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            data = json.load(f)
        for r in data.get("results", []):
            exclude.add(r["filename"])

    filenames = sample_eval_filenames(num_samples, exclude)
    logger.info(f"Eval: sampled {len(filenames)} held-out images")

    with tempfile.TemporaryDirectory() as tmp_dir:
        file_pairs = [(fn, os.path.join(tmp_dir, fn)) for fn in filenames]
        batch_size = 50
        for i in tqdm(range(0, len(file_pairs), batch_size), desc="Downloading eval images"):
            batch = file_pairs[i : i + batch_size]
            for attempt in range(3):
                try:
                    download_bucket_files(BUCKET_REPO, files=batch, token=HF_TOKEN)
                    break
                except Exception as e:
                    logger.warning(f"Download attempt {attempt+1}/3 failed: {e}")
                    time.sleep(2 ** attempt)

        results = []
        for fn in tqdm(filenames, desc="Running eval"):
            path = os.path.join(tmp_dir, fn)
            if not os.path.exists(path):
                continue
            try:
                img = load_image_safely(path)
                yolo_results = yolo_model(img, verbose=False)
                probs = yolo_results[0].probs
                pred_name = yolo_results[0].names[probs.top1]
                pred = LABEL_TO_QUARTERS.get(pred_name, 0)
                if pred != 0:
                    results.append({"filename": fn, "prediction": pred, "image": img})
            except Exception as e:
                logger.warning(f"Eval failed on {fn}: {e}")

        logger.info(f"Eval: {len(results)} images with corrections out of {len(filenames)}")

        doc = SimpleDocTemplate(
            output_path, pagesize=letter,
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        )
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("YOLO Orientation Predictions (Held-Out Images)", styles["Title"]))
        elements.append(Paragraph(
            f"{len(results)} corrected images from {len(filenames)} held-out samples",
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

                caption = Paragraph(f"<b>{entry['filename']}</b> — correction: {label}", styles["Normal"])
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
    parser.add_argument("--val-split", type=float, default=0.15)
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

    # Build dataset structure
    dataset_dir = os.path.join(args.output_dir, "dataset")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    logger.info("Building classification dataset...")
    build_dataset(labels, image_dir, dataset_dir, val_split=args.val_split, no_synthetic=args.no_synthetic)

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
        generate_eval_pdf(eval_model, args.labels, args.eval_pdf)


if __name__ == "__main__":
    main()
