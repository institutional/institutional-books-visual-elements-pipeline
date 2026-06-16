"""
Manual orientation labeling GUI (web-based with Gradio).

Downloads images from the HF bucket and presents them one at a time.
Rotate the image until it looks upright, then confirm.
Results are saved in the same format as training_labels.json.

Usage:
    python orientation_tests/manual_labeler.py [--sample-size 100] [--output orientation_tests/manual_labels.json]

Controls:
    Left arrow / A  : Rotate 90° counter-clockwise
    Right arrow / D : Rotate 90° clockwise
    Down arrow / S  : Rotate 180°
    Enter / Space   : Confirm as upright (save current orientation)
    U               : Undo last rotation (reset to original)
"""

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import download_bucket_files
from loguru import logger
from PIL import Image, ImageOps
from datasets import load_dataset

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
BUCKET_REPO = "institutional/institutional-books-hl-visual-elements-images"
DATASET_REPO = "institutional/institutional-books-hl-visual-elements"
ALLOWED_SPLITS = ["image_illustration", "chart_graph"]

QUARTERS_TO_LABEL = {
    0: "upright",
    1: "rotate_90_clockwise",
    2: "rotate_180",
    3: "rotate_90_counterclockwise",
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


def sample_filenames_from_dataset(sample_size: int) -> list[str]:
    logger.info(f"Sampling {sample_size} filenames from dataset...")
    per_split = sample_size // len(ALLOWED_SPLITS)
    filenames = []

    for split in ALLOWED_SPLITS:
        ds = load_dataset(DATASET_REPO, split=split, streaming=True, token=HF_TOKEN)
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

    random.shuffle(filenames)
    return filenames[:sample_size]


DOWNLOAD_TIMEOUT = 60
DOWNLOAD_MAX_RETRIES = 3


def download_images(filenames: list[str], image_dir: str):
    logger.info(f"Downloading {len(filenames)} images...")
    file_pairs = [(fn, os.path.join(image_dir, fn)) for fn in filenames]
    batch_size = 200
    for i in range(0, len(file_pairs), batch_size):
        batch = file_pairs[i : i + batch_size]
        for attempt in range(DOWNLOAD_MAX_RETRIES):
            try:
                start = time.time()
                download_bucket_files(BUCKET_REPO, files=batch, token=HF_TOKEN)
                elapsed = time.time() - start
                if elapsed > DOWNLOAD_TIMEOUT:
                    logger.warning(f"Batch {i // batch_size} took {elapsed:.0f}s")
                break
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1}/{DOWNLOAD_MAX_RETRIES} failed: {e}")
                time.sleep(2 ** attempt)
                if attempt == DOWNLOAD_MAX_RETRIES - 1:
                    logger.error(f"Skipping batch at index {i} after {DOWNLOAD_MAX_RETRIES} retries")
    logger.info("Download complete")


def apply_rotation(img: Image.Image, quarters_cw: int) -> Image.Image:
    degrees_map = {0: 0, 1: 270, 2: 180, 3: 90}
    q = quarters_cw % 4
    if q == 0:
        return img
    return img.rotate(degrees_map[q], expand=True)


def main():
    parser = argparse.ArgumentParser(description="Manual orientation labeling GUI")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output", type=str, default="orientation_tests/manual_labels.json")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Reuse an existing image directory (skip download)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file, skipping already-labeled images")
    args = parser.parse_args()

    # Load already-labeled filenames to exclude
    already_labeled = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing_data = json.load(f)
        already_labeled = {r["filename"] for r in existing_data.get("results", [])}
        logger.info(f"Excluding {len(already_labeled)} already-labeled images")

    if args.image_dir:
        image_dir = args.image_dir
        filenames = [f for f in sorted(os.listdir(image_dir)) if f not in already_labeled]
        logger.info(f"Reusing image dir: {image_dir} ({len(filenames)} unlabeled images)")
    else:
        image_dir = tempfile.mkdtemp(prefix="orientation_labeler_")
        logger.info(f"Image dir: {image_dir}")
        filenames = sample_filenames_from_dataset(args.sample_size)
        filenames = [f for f in filenames if f not in already_labeled]
        download_images(filenames, image_dir)

    results = []
    start_idx = 0

    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        results = existing.get("results", [])
        labeled_set = {r["filename"] for r in results}
        # Skip past already-labeled images
        for i, fn in enumerate(filenames):
            if fn not in labeled_set:
                start_idx = i
                break
        else:
            start_idx = len(filenames)
        logger.info(f"Resuming: {len(results)} already labeled, starting at index {start_idx}")

    state = {"idx": start_idx, "rotation": 0}

    def get_current_image():
        if state["idx"] >= len(filenames):
            return None, f"Done! Labeled {len(results)} images."
        filename = filenames[state["idx"]]
        path = os.path.join(image_dir, filename)
        try:
            img = load_image_safely(path)
        except Exception:
            state["idx"] += 1
            return get_current_image()
        img = apply_rotation(img, state["rotation"])
        label = QUARTERS_TO_LABEL[state["rotation"] % 4]
        info = f"[{state['idx'] + 1}/{len(filenames)}] {filename} — applied: {label}"
        return img, info

    def rotate_cw():
        state["rotation"] = (state["rotation"] + 1) % 4
        return get_current_image()

    def rotate_ccw():
        state["rotation"] = (state["rotation"] - 1) % 4
        return get_current_image()

    def rotate_180():
        state["rotation"] = (state["rotation"] + 2) % 4
        return get_current_image()

    def reset():
        state["rotation"] = 0
        return get_current_image()

    def save_and_next():
        if state["idx"] >= len(filenames):
            return get_current_image()

        filename = filenames[state["idx"]]
        original_orientation = state["rotation"] % 4
        label = QUARTERS_TO_LABEL[original_orientation]

        results.append({
            "filename": filename,
            "manual_label": label,
        })

        save_to_disk(results, args.output)

        state["idx"] += 1
        state["rotation"] = 0
        return get_current_image()

    def back():
        if state["idx"] > 0:
            state["idx"] -= 1
            if results and results[-1]["filename"] == filenames[state["idx"]]:
                results.pop()
                save_to_disk(results, args.output)
            state["rotation"] = 0
        return get_current_image()

    def skip():
        state["idx"] += 1
        state["rotation"] = 0
        return get_current_image()

    def save_to_disk(results_list, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        from collections import Counter
        preds = Counter(r["manual_label"] for r in results_list)
        output = {
            "summary": {
                "total_labeled": len(results_list),
                "orientation_distribution": dict(preds),
            },
            "results": results_list,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    keyboard_js = """
    () => {
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const key = e.key.toLowerCase();
            const btns = document.querySelectorAll('button');
            const clickBtn = (text) => {
                for (const b of btns) {
                    if (b.textContent.includes(text)) { b.click(); break; }
                }
            };
            if (key === 'a' || key === 'arrowleft') { clickBtn('CCW'); e.preventDefault(); }
            else if (key === 'd' || key === 'arrowright') { clickBtn('CW (D)'); e.preventDefault(); }
            else if (key === 's' || key === 'arrowdown') { clickBtn('180'); e.preventDefault(); }
            else if (key === 'u') { clickBtn('Reset'); e.preventDefault(); }
            else if (key === 'enter' || key === ' ') { clickBtn('Save & Next'); e.preventDefault(); }
        });
    }
    """

    with gr.Blocks(title="Orientation Labeler") as app:
        gr.Markdown("# Orientation Labeler\nRotate the image until it looks upright, then click **Save & Next**.")

        info = gr.Textbox(value="Loading...", label="Status", interactive=False)
        image = gr.Image(value=None, label="Image", type="pil", height=500)

        app.load(get_current_image, outputs=[image, info])

        with gr.Row():
            btn_back = gr.Button("← Back", variant="secondary")
            btn_ccw = gr.Button("↶ CCW (A)", variant="secondary")
            btn_180 = gr.Button("180° (S)", variant="secondary")
            btn_cw = gr.Button("↷ CW (D)", variant="secondary")
            btn_reset = gr.Button("Reset (U)", variant="secondary")
            btn_skip = gr.Button("Skip", variant="secondary")
            btn_save = gr.Button("Save & Next (Enter)", variant="primary")

        btn_back.click(back, outputs=[image, info])
        btn_ccw.click(rotate_ccw, outputs=[image, info])
        btn_cw.click(rotate_cw, outputs=[image, info])
        btn_180.click(rotate_180, outputs=[image, info])
        btn_reset.click(reset, outputs=[image, info])
        btn_skip.click(skip, outputs=[image, info])
        btn_save.click(save_and_next, outputs=[image, info])

        app.load(None, js=keyboard_js)

    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
