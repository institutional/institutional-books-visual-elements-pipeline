"""
Train an EfficientNet-V2-S orientation classifier using VLM-generated labels.

Reads training_labels.json (produced by generate_training_labels.py), downloads
the images, creates all 4 rotations per image as training samples, and fine-tunes
an EfficientNet-V2-S for 4-class orientation prediction.

Usage:
    python orientation_tests/train_orientation_model.py [--labels orientation_tests/training_labels.json] [--epochs 20]
"""

import argparse
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
from dotenv import load_dotenv
from huggingface_hub import download_bucket_files
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
BUCKET_REPO = "institutional/institutional-books-hl-visual-elements-images"

IMAGE_SIZE = 384
NUM_CLASSES = 4
CLASS_MAP = {
    0: "upright",
    1: "rotated_90_clockwise",
    2: "rotated_180",
    3: "rotated_90_counterclockwise",
}

VLM_LABEL_TO_QUARTERS = {
    "upright": 0,
    "rotated_90_clockwise": 1,
    "rotated_180": 2,
    "rotated_90_counterclockwise": 3,
}

# Correction: rotate by this many quarter-turns CW to make upright
CORRECTION_QUARTERS = {0: 0, 1: 3, 2: 2, 3: 1}


class OrientationDataset(Dataset):
    def __init__(self, image_dir: str, labels: list, transform=None):
        self.samples = []
        self.transform = transform

        for entry in labels:
            filename = entry["filename"]
            path = os.path.join(image_dir, filename)
            if not os.path.exists(path):
                continue
            original_orientation = entry["original_orientation"]
            # Create all 4 rotations from the corrected (upright) image
            for rot_class in range(4):
                self.samples.append((path, original_orientation, rot_class))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, original_orientation, rot_class = self.samples[idx]
        try:
            img = load_image_safely(path)
        except Exception:
            # Return a blank image for corrupt files
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
            rot_class = 0
        else:
            correction = CORRECTION_QUARTERS[original_orientation]
            if correction:
                img = apply_rotation(img, correction)
            if rot_class:
                img = apply_rotation(img, rot_class)
        if self.transform:
            img = self.transform(img)
        return img, rot_class


def load_image_safely(path: str) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGB", "L"):
            return img.convert("RGB")
        rgba_img = img.convert("RGBA")
        background = Image.new("RGB", rgba_img.size, (255, 255, 255))
        background.paste(rgba_img, mask=rgba_img)
        return background


def apply_rotation(img: Image.Image, rotation_class: int) -> Image.Image:
    degrees_map = {0: 0, 1: 270, 2: 180, 3: 90}
    return img.rotate(degrees_map[rotation_class], expand=True)


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


DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 3


def download_training_images(filenames: list[str], image_dir: str):
    logger.info(f"Downloading {len(filenames)} images for training...")
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


def main():
    parser = argparse.ArgumentParser(description="Train orientation EfficientNet model")
    parser.add_argument("--labels", type=str, default="orientation_tests/training_labels.json")
    parser.add_argument("--output-dir", type=str, default="orientation_tests/model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Use only the first N labels")
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
        label_field = r.get("vlm_prediction") or r.get("manual_label")
        if not label_field or label_field not in VLM_LABEL_TO_QUARTERS:
            continue
        quarters = VLM_LABEL_TO_QUARTERS[label_field]
        labels.append({
            "filename": r["filename"],
            "original_orientation": quarters,
        })
    logger.info(f"Using {len(labels)} labels")

    if not labels:
        logger.error("No valid labels found. Run generate_training_labels.py first.")
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
        download_training_images(to_download, image_dir)
    else:
        logger.info("All images already downloaded")

    # Create datasets
    train_dataset = OrientationDataset(image_dir, train_labels, transform=get_train_transform())
    val_dataset = OrientationDataset(image_dir, val_labels, transform=get_val_transform())
    logger.info(f"Train samples (4 rotations each): {len(train_dataset)}")
    logger.info(f"Val samples (4 rotations each): {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Compute class weights from label distribution (before 4x rotation augmentation)
    from collections import Counter
    orientation_counts = Counter(l["original_orientation"] for l in labels)
    total_labels = sum(orientation_counts.values())
    class_weights = torch.tensor(
        [total_labels / (NUM_CLASSES * orientation_counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Build model
    model = build_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
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
            save_path = output_dir / f"orientation_model_best_{val_acc:.4f}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"  New best model saved: {save_path}")

    # Save final model
    final_path = output_dir / "orientation_model_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved: {final_path}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
