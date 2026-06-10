"""
Train an EfficientNet-V2-S orientation classifier.

Reads manual_labels.json, downloads the images, and trains a 4-class orientation
model. Predictions represent the correction needed to make an image upright.

Usage:
    python orientation_tests/train_orientation_model.py [--labels orientation_tests/manual_labels.json] [--epochs 20]
    python orientation_tests/train_orientation_model.py --no-synthetic --eval-pdf orientation_tests/eval.pdf
"""

import argparse
import io
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import download_bucket_files
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
BUCKET_REPO = "institutional/institutional-books-hl-visual-elements-images"
DATASET_REPO = "institutional/institutional-books-hl-visual-elements"
ALLOWED_SPLITS = ["image_illustration", "chart_graph"]

IMAGE_SIZE = 384
NUM_CLASSES = 4
CLASS_MAP = {
    0: "upright",
    1: "rotated_90_clockwise",
    2: "rotated_180",
    3: "rotated_90_counterclockwise",
}

LABEL_TO_QUARTERS = {
    "upright": 0,
    "rotated_90_clockwise": 1,
    "rotated_180": 2,
    "rotated_90_counterclockwise": 3,
}

DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 3


class OrientationDataset(Dataset):
    def __init__(self, image_dir: str, labels: list, transform=None, no_synthetic=False):
        self.samples = []
        self.transform = transform

        for entry in labels:
            filename = entry["filename"]
            path = os.path.join(image_dir, filename)
            if not os.path.exists(path):
                continue
            correction_quarters = entry["correction_quarters"]
            if no_synthetic:
                self.samples.append((path, 0, correction_quarters))
            else:
                for rot_class in range(4):
                    self.samples.append((path, correction_quarters, rot_class))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, correction_quarters, rot_class = self.samples[idx]
        try:
            img = load_image_safely(path)
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            rot_class = 0
        else:
            if correction_quarters:
                img = apply_rotation(img, correction_quarters)
            if rot_class:
                img = apply_rotation(img, rot_class)
        if self.transform:
            img = self.transform(img)
        # Label = correction needed to make upright
        label = (4 - rot_class) % 4
        return img, label


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


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(device: torch.device) -> nn.Module:
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, NUM_CLASSES),
    )
    return model.to(device)


def download_images(filenames: list[str], image_dir: str):
    logger.info(f"Downloading {len(filenames)} images...")
    file_pairs = [(fn, os.path.join(image_dir, fn)) for fn in filenames]

    batch_size = 100
    for i in tqdm(range(0, len(file_pairs), batch_size), desc="Downloading"):
        batch = file_pairs[i : i + batch_size]
        for attempt in range(DOWNLOAD_MAX_RETRIES):
            try:
                start = time.time()
                download_bucket_files(BUCKET_REPO, files=batch, token=HF_TOKEN)
                elapsed = time.time() - start
                if elapsed > DOWNLOAD_TIMEOUT:
                    logger.warning(f"Download took {elapsed:.0f}s (attempt {attempt + 1})")
                break
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1}/{DOWNLOAD_MAX_RETRIES} failed: {e}")
                time.sleep(2 ** attempt)
                if attempt == DOWNLOAD_MAX_RETRIES - 1:
                    logger.error(f"Skipping batch at {i}")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


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


def generate_eval_pdf(model, device, labels_file: str, output_path: str, num_samples: int = 500):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepTogether,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet

    # Build exclusion set from training labels
    exclude = set()
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            data = json.load(f)
        for r in data.get("results", []):
            exclude.add(r["filename"])

    filenames = sample_eval_filenames(num_samples, exclude)
    logger.info(f"Eval: sampled {len(filenames)} held-out images")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download
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

        # Run inference
        transform = get_val_transform()
        results = []
        model.eval()

        for fn in tqdm(filenames, desc="Running eval"):
            path = os.path.join(tmp_dir, fn)
            if not os.path.exists(path):
                continue
            try:
                img = load_image_safely(path)
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor)
                pred = output.argmax(dim=1).item()
                if pred != 0:
                    results.append({"filename": fn, "prediction": pred, "image": img})
            except Exception as e:
                logger.warning(f"Eval failed on {fn}: {e}")

        logger.info(f"Eval: {len(results)} images with corrections out of {len(filenames)}")

        # Build PDF
        doc = SimpleDocTemplate(
            output_path, pagesize=letter,
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        )
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Model Orientation Predictions (Held-Out Images)", styles["Title"]))
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
                label = CLASS_MAP[pred_class]
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
    parser = argparse.ArgumentParser(description="Train orientation EfficientNet model")
    parser.add_argument("--labels", type=str, default="orientation_tests/manual_labels.json")
    parser.add_argument("--output-dir", type=str, default="orientation_tests/model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Use only the first N labels")
    parser.add_argument("--no-synthetic", action="store_true",
                        help="Train on raw images with their labels as-is, no correction + 4-rotation augmentation")
    parser.add_argument("--eval-pdf", type=str, default=None,
                        help="After training, generate an evaluation PDF at this path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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

    # Split into train/val
    random.shuffle(labels)
    val_size = int(len(labels) * args.val_split)
    val_labels = labels[:val_size]
    train_labels = labels[val_size:]
    logger.info(f"Train: {len(train_labels)}, Val: {len(val_labels)}")

    # Download images
    all_filenames = list(set(r["filename"] for r in labels))
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    existing = set(os.listdir(image_dir))
    to_download = [fn for fn in all_filenames if fn not in existing]
    if to_download:
        download_images(to_download, image_dir)
    else:
        logger.info("All images already downloaded")

    # Create datasets
    train_dataset = OrientationDataset(image_dir, train_labels, transform=get_train_transform(), no_synthetic=args.no_synthetic)
    val_dataset = OrientationDataset(image_dir, val_labels, transform=get_val_transform(), no_synthetic=args.no_synthetic)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Build model
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    best_model_path = None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} — "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} — "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = output_dir / f"orientation_model_best_{val_acc:.4f}.pth"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  New best model saved: {best_model_path}")

    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")

    # Generate eval PDF if requested
    if args.eval_pdf and best_model_path:
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        generate_eval_pdf(model, device, args.labels, args.eval_pdf)


if __name__ == "__main__":
    main()
