"""
Generate orientation training labels using Qwen2.5-VL-7B.

For each sampled image, rotates it by a random known amount, then asks the VLM
what rotation was applied. This creates ground-truth labels from intentionally
rotated images rather than relying on the VLM to assess "natural" orientation
(which is ambiguous for many book elements).

Strategy:
  1. Sample images from the HF bucket
  2. For each image, randomly rotate by 0°, 90°, 180°, or 270°
  3. Ask the VLM: "What rotation does this image need to be corrected?"
  4. Save (filename, applied_rotation, vlm_prediction) as the training set

Images where the VLM agrees with the known rotation become high-quality labels.
Images where it's ambiguous (e.g. symmetric patterns) and the VLM says "correct"
regardless of rotation get labeled as "correct" — teaching the EfficientNet to
not flag them.

Usage:
    python orientation_tests/generate_training_labels.py [--sample-size 5000] [--output orientation_tests/training_labels.json]
"""

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import download_bucket_files
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
BUCKET_REPO = "institutional/institutional-books-hl-visual-elements-images"
VLM_MODEL = "Qwen/Qwen3-VL-32B-Instruct"

DATASET_REPO = "institutional/institutional-books-hl-visual-elements"
ALLOWED_SPLITS = ["image_illustration", "chart_graph"]

ORIENTATION_CLASSES = ["upright", "rotated_90_clockwise", "rotated_180", "rotated_90_counterclockwise"]

PROMPT = "Is this image upright or rotated? Reply with one word: upright, rotated_90_clockwise, rotated_180, or rotated_90_counterclockwise."


DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 3


def download_with_retry(file_pairs: list, max_retries: int = DOWNLOAD_MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            start = time.time()
            download_bucket_files(
                BUCKET_REPO,
                files=file_pairs,
                token=HF_TOKEN,
            )
            elapsed = time.time() - start
            if elapsed > DOWNLOAD_TIMEOUT:
                logger.warning(f"Download took {elapsed:.0f}s (attempt {attempt + 1})")
            return True
        except Exception as e:
            logger.warning(f"Download failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    return False


def load_vlm(device: torch.device):
    logger.info(f"Loading {VLM_MODEL}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL, token=HF_TOKEN)
    logger.info("VLM loaded")
    return model, processor


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


def classify_orientation(
    model, processor, image_path: str, device: torch.device
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=500)

    input_len = inputs.input_ids.shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True).strip().lower()

    # Strip thinking blocks if present
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    for label in ORIENTATION_CLASSES:
        if label in response:
            return label

    return "upright"


def sample_filenames_from_dataset(sample_size: int) -> list[str]:
    """Sample image filenames from the HF dataset, restricted to allowed classes."""
    logger.info(f"Sampling {sample_size} filenames from dataset (splits: {ALLOWED_SPLITS})...")
    per_split = sample_size // len(ALLOWED_SPLITS)
    filenames = []

    for split in ALLOWED_SPLITS:
        logger.info(f"Loading split '{split}'...")
        ds = load_dataset(
            DATASET_REPO, split=split, streaming=True, token=HF_TOKEN
        )

        reservoir = []
        for i, row in enumerate(ds):
            stem = Path(row["page_filename_src"]).stem
            fn = f"{row['barcode_src']}_{stem}_{row['id']}.webp"
            if i < per_split:
                reservoir.append(fn)
            else:
                j = random.randint(0, i)
                if j < per_split:
                    reservoir[j] = fn
            if i >= per_split * 10:
                break

        filenames.extend(reservoir)
        logger.info(f"  Sampled {len(reservoir)} from '{split}'")

    random.shuffle(filenames)
    return filenames[:sample_size]


def main():
    parser = argparse.ArgumentParser(description="Generate orientation training labels with VLM")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--output", type=str, default="orientation_tests/training_labels.json")
    parser.add_argument("--batch-download-size", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_vlm(device)

    sampled_filenames = sample_filenames_from_dataset(args.sample_size)
    logger.info(f"Processing {len(sampled_filenames)} images")

    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for batch_start in tqdm(
            range(0, len(sampled_filenames), args.batch_download_size),
            desc="Processing batches",
        ):
            batch = sampled_filenames[batch_start : batch_start + args.batch_download_size]

            file_pairs = [
                (filename, os.path.join(tmp_dir, filename))
                for filename in batch
            ]

            if not download_with_retry(file_pairs):
                logger.error(f"Skipping batch at {batch_start} after {DOWNLOAD_MAX_RETRIES} retries")
                continue

            # Prepare all rotated images for this download batch
            inference_items = []
            for filename, local_path in file_pairs:
                try:
                    original = load_image_safely(local_path)
                    applied_rotation = random.randint(0, 3)
                    rotated = apply_rotation(original, applied_rotation)
                    rotated_path = os.path.join(tmp_dir, f"rotated_{filename}")
                    rotated.save(rotated_path, format="PNG")
                    inference_items.append({
                        "filename": filename,
                        "rotated_path": rotated_path,
                        "applied_rotation": applied_rotation,
                    })
                except Exception as e:
                    logger.warning(f"Failed to prepare {filename}: {e}")
                    results.append({"filename": filename, "error": str(e)})
                finally:
                    if os.path.exists(local_path):
                        os.remove(local_path)

            # Run VLM inference one image at a time
            for item in tqdm(inference_items, desc="VLM inference", leave=False):
                try:
                    vlm_label = classify_orientation(model, processor, item["rotated_path"], device)
                except Exception as e:
                    logger.warning(f"Inference failed for {item['filename']}: {e}")
                    vlm_label = "correct"

                expected_label = ORIENTATION_CLASSES[item["applied_rotation"]]
                results.append({
                    "filename": item["filename"],
                    "applied_rotation": item["applied_rotation"],
                    "applied_label": expected_label,
                    "vlm_prediction": vlm_label,
                    "agrees": vlm_label == expected_label,
                })
                os.remove(item["rotated_path"])

            if batch_start % 500 == 0 and results:
                save_results(results, args.output)

    save_results(results, args.output)
    print_summary(results)


def save_results(results: list, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    successful = [r for r in results if "error" not in r]
    agreement = sum(1 for r in successful if r["agrees"])
    summary = {
        "total_processed": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "agreement_rate": round(agreement / len(successful) * 100, 2) if successful else 0,
    }

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)


def print_summary(results: list):
    successful = [r for r in results if "error" not in r]
    agreement = sum(1 for r in successful if r["agrees"])
    logger.info(f"Total: {len(results)}, Successful: {len(successful)}")
    logger.info(f"VLM agreement with known rotation: {agreement}/{len(successful)} ({agreement/len(successful)*100:.1f}%)")

    by_class = {}
    for r in successful:
        label = r["applied_label"]
        by_class.setdefault(label, {"total": 0, "agree": 0})
        by_class[label]["total"] += 1
        if r["agrees"]:
            by_class[label]["agree"] += 1

    for label, counts in by_class.items():
        rate = counts["agree"] / counts["total"] * 100
        logger.info(f"  {label}: {counts['agree']}/{counts['total']} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
