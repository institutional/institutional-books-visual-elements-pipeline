"""
Train a YOLO26n-cls orientation classifier using manual/VLM labels.

Reads labels JSON, organizes images into class folders (with all 4 rotations
per image as augmentation), and fine-tunes YOLO26n-cls for 4-class orientation.

Usage:
    python orientation_tests/train_yolo_cls.py [--labels orientation_tests/manual_labels.json] [--epochs 50]
"""

import argparse
import json
import os
import random
import shutil
import tempfile
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

CORRECTION_QUARTERS = {0: 0, 1: 3, 2: 2, 3: 1}

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


def build_dataset(labels: list, image_dir: str, dataset_dir: str, val_split: float = 0.15):
    """Organize images into train/val class folders with all 4 rotations."""
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

            original_quarters = entry["original_orientation"]
            correction = CORRECTION_QUARTERS[original_quarters]
            if correction:
                img = apply_rotation(img, correction)

            stem = Path(filename).stem
            for rot_class in range(4):
                rotated = apply_rotation(img, rot_class) if rot_class else img
                cls_name = CLASS_NAMES[rot_class]
                out_path = os.path.join(dataset_dir, split_name, cls_name, f"{stem}_r{rot_class}.jpg")
                rotated.save(out_path, "JPEG", quality=90)

        if skipped:
            logger.warning(f"{split_name}: skipped {skipped} images (missing or corrupt)")

    for split_name in splits:
        for cls_name in CLASS_NAMES:
            count = len(os.listdir(os.path.join(dataset_dir, split_name, cls_name)))
            logger.info(f"  {split_name}/{cls_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n-cls orientation model")
    parser.add_argument("--labels", type=str, default="orientation_tests/manual_labels.json")
    parser.add_argument("--output-dir", type=str, default="orientation_tests/model")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Reuse existing image directory (skip download)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    with open(args.labels) as f:
        data = json.load(f)

    results = data["results"][:args.max_samples] if args.max_samples else data["results"]
    labels = []
    for r in results:
        if "error" in r:
            continue
        label_field = r.get("vlm_prediction") or r.get("manual_label")
        if not label_field or label_field not in LABEL_TO_QUARTERS:
            continue
        labels.append({
            "filename": r["filename"],
            "original_orientation": LABEL_TO_QUARTERS[label_field],
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
    build_dataset(labels, image_dir, dataset_dir, val_split=args.val_split)

    # Train
    model = YOLO("yolo26m-cls.pt")
    results = model.train(
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

    logger.info(f"Training complete. Results saved to {args.output_dir}/yolo26n_orientation/")


if __name__ == "__main__":
    main()
