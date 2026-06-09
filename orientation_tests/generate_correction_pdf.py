"""
Generate a PDF showing original bucket images alongside VLM-corrected versions.

Uses the inferred original orientation (from VLM prediction minus applied rotation)
to show before/after correction on a random sample.

Usage:
    python orientation_tests/generate_correction_pdf.py [--input orientation_tests/training_labels.json] [--output orientation_tests/corrections.pdf]
"""

import argparse
import io
import json
import os
import random
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import download_bucket_files
from loguru import logger
from PIL import Image, ImageOps
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    KeepTogether,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet
from tqdm import tqdm

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
BUCKET_REPO = "institutional/institutional-books-hl-visual-elements-images"

DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 3

VLM_LABEL_TO_QUARTERS = {
    "upright": 0,
    "rotated_90_clockwise": 1,
    "rotated_180": 2,
    "rotated_90_counterclockwise": 3,
}

QUARTERS_TO_LABEL = {
    0: "upright",
    1: "rotated_90_clockwise",
    2: "rotated_180",
    3: "rotated_90_counterclockwise",
}

def load_image_safely(path: str) -> Image.Image:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGB", "L"):
            return img.convert("RGB")
        rgba_img = img.convert("RGBA")
        background = Image.new("RGB", rgba_img.size, (255, 255, 255))
        background.paste(rgba_img, mask=rgba_img)
        return background


def image_to_bytesio(img: Image.Image, max_dim: int = 300) -> io.BytesIO:
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def download_with_retry(file_pairs: list) -> bool:
    for attempt in range(DOWNLOAD_MAX_RETRIES):
        try:
            start = time.time()
            download_bucket_files(BUCKET_REPO, files=file_pairs, token=HF_TOKEN)
            elapsed = time.time() - start
            if elapsed > DOWNLOAD_TIMEOUT:
                logger.warning(f"Download took {elapsed:.0f}s (attempt {attempt + 1})")
            return True
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1}/{DOWNLOAD_MAX_RETRIES} failed: {e}")
            time.sleep(2 ** attempt)
    return False


def apply_rotation(img: Image.Image, quarters_cw: int) -> Image.Image:
    """Rotate image by N quarter-turns clockwise."""
    # PIL .rotate() goes CCW, so 1 quarter CW = rotate(-90) = rotate(270)
    degrees_map = {0: 0, 1: 270, 2: 180, 3: 90}
    q = quarters_cw % 4
    if q == 0:
        return img
    return img.rotate(degrees_map[q], expand=True)


def build_pdf(entries: list, tmp_dir: str, output_path: str, source: str = "vlm"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    elements = []

    if source == "manual":
        title = Paragraph("Manual Orientation Labels Verification", styles["Title"])
    else:
        title = Paragraph("VLM Orientation Labels Verification", styles["Title"])
    subtitle = Paragraph(
        f"{len(entries)} random images: original (bucket) vs corrected",
        styles["Normal"],
    )
    elements.append(title)
    elements.append(subtitle)
    elements.append(Spacer(1, 0.3 * inch))

    img_width = 2.8 * inch
    img_height = 2.8 * inch

    for entry in tqdm(entries, desc="Building PDF pages"):
        filename = entry["filename"]
        local_path = os.path.join(tmp_dir, filename)

        if not os.path.exists(local_path):
            continue

        try:
            original = load_image_safely(local_path)

            label = entry["vlm_prediction"]
            quarters = VLM_LABEL_TO_QUARTERS[label]

            if source == "manual":
                # Manual label = rotation user applied to make it upright
                correction = quarters
            else:
                # VLM label = how the image is oriented → apply opposite
                correction = (4 - quarters) % 4

            corrected = apply_rotation(original, correction)

            orig_buf = image_to_bytesio(original.copy())
            cor_buf = image_to_bytesio(corrected.copy())

            orig_img = RLImage(orig_buf, width=img_width, height=img_height, kind="proportional")
            cor_img = RLImage(cor_buf, width=img_width, height=img_height, kind="proportional")

            header_table = Table(
                [[
                    Paragraph("<b>Original</b>", styles["Normal"]),
                    Paragraph("<b>Corrected</b>", styles["Normal"]),
                ]],
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
                f"<b>{filename}</b> — label: {label}",
                styles["Normal"],
            )

            block = KeepTogether([
                caption,
                Spacer(1, 0.1 * inch),
                header_table,
                img_table,
                Spacer(1, 0.3 * inch),
            ])
            elements.append(block)

        except Exception as e:
            logger.warning(f"Failed to add {filename} to PDF: {e}")

    doc.build(elements)
    logger.info(f"PDF saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate correction PDF from VLM labels")
    parser.add_argument("--input", type=str, default="orientation_tests/training_labels.json")
    parser.add_argument("--output", type=str, default="orientation_tests/corrections.pdf")
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results = data["results"]
    valid = [r for r in results if "error" not in r]

    # Detect source type
    source = "manual" if valid and "manual_label" in valid[0] else "vlm"

    # Normalize to vlm_prediction field for downstream
    for r in valid:
        if "vlm_prediction" not in r and "manual_label" in r:
            r["vlm_prediction"] = r["manual_label"]

    # Filter to those needing correction
    needs_correction = [r for r in valid if r["vlm_prediction"] != "upright"]
    logger.info(f"Found {len(needs_correction)} images needing correction out of {len(valid)} total")

    # Random sample
    entries = random.sample(needs_correction, min(args.num_samples, len(needs_correction)))

    with tempfile.TemporaryDirectory() as tmp_dir:
        filenames = [r["filename"] for r in entries]
        file_pairs = [(fn, os.path.join(tmp_dir, fn)) for fn in filenames]

        logger.info(f"Downloading {len(file_pairs)} images...")
        batch_size = 50
        for i in tqdm(range(0, len(file_pairs), batch_size), desc="Downloading"):
            batch = file_pairs[i : i + batch_size]
            if not download_with_retry(batch):
                logger.error(f"Failed to download batch at {i}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        build_pdf(entries, tmp_dir, str(output_path), source=source)


if __name__ == "__main__":
    main()
