"""
Visual Elements Viewer - Interactive walkthrough of pipeline stages.
Deployed as a HuggingFace Space with pre-exported static data.
"""

import json
import base64
from pathlib import Path
from collections import defaultdict

import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

CLASSIFICATION_CLASS_DICT = {
    "Other": "Other",
    "Image or Illustration": "Image/Illustration",
    "Ex Libris or Decorative": "Ex Libris/Decorative",
    "Music": "Music",
    "Chart or Graph": "Chart/Graph",
    "Artifact": "Artifact",
}

CAPTION_CLASSES_EXCLUDED = ["Ex Libris or Decorative", "Artifact"]

CLASS_COLORS = {
    "Image/Illustration": (46, 204, 113),
    "Ex Libris/Decorative": (230, 126, 34),
    "Music": (52, 152, 219),
    "Chart/Graph": (155, 89, 182),
    "Artifact": (241, 196, 15),
    "Other": (149, 165, 166),
}

DATA_DIR = Path("data")
_volume_cache = {}
_images_cache = {}

DETECT_REPO = "institutional/institutional-books-visual-elements-detection-yolo26n"
CLASSIFY_REPO = "institutional/institutional-books-visual-elements-classification-yolo26s-cls"
DATA_REPO = "institutional/institutional-books-hl-visual-elements-demo"


def _load_models():
    import os
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    detect_path = DATA_DIR / "detect.pt"
    classify_path = DATA_DIR / "classify.pt"
    if not detect_path.exists():
        detect_path = Path(hf_hub_download(DETECT_REPO, "weights/best.pt", token=token))
    if not classify_path.exists():
        classify_path = Path(hf_hub_download(CLASSIFY_REPO, "weights/best.pt", token=token))
    return YOLO(str(detect_path)), YOLO(str(classify_path))


DETECT_MODEL, CLASSIFY_MODEL = _load_models()

CLASSIFY_NAMES = {
    0: "Artifact",
    1: "Chart or Graph",
    2: "Ex Libris or Decorative",
    3: "Image or Illustration",
    4: "Music",
}


def run_inference(image):
    """Run detection + classification on an uploaded image."""
    if image is None:
        return None, ""

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    det_results = DETECT_MODEL.predict(img_bgr, imgsz=800, conf=0.25, verbose=False)
    boxes = det_results[0].boxes

    if len(boxes) == 0:
        return img_rgb, "No visual elements detected."

    pil_img = image.copy()
    draw = ImageDraw.Draw(pil_img)
    img_w, img_h = pil_img.size
    font_size = max(18, min(40, img_w // 50))
    line_width = max(3, min(8, img_w // 400))
    font = _get_font(font_size)

    results_rows = []
    results_rows.append("| # | Class | Det % | Cls % | Size |")
    results_rows.append("|---|---|---|---|---|")

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        det_conf = float(box.conf[0])

        crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        cls_results = CLASSIFY_MODEL.predict(crop, imgsz=480, verbose=False)
        probs = cls_results[0].probs
        cls_idx = int(probs.top1)
        cls_conf = float(probs.top1conf)
        class_name = CLASSIFY_NAMES.get(cls_idx, "Other")

        display_name = CLASSIFICATION_CLASS_DICT.get(class_name, class_name)
        color = CLASS_COLORS.get(display_name, (149, 165, 166))

        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=color, width=line_width)

        label = f"{int(det_conf * 100)}% {display_name}"
        lbox = draw.textbbox((0, 0), label, font=font)
        lw, lh = lbox[2] - lbox[0], lbox[3] - lbox[1]
        label_y = int(y1) - lh - 6
        if label_y < 0:
            label_y = int(y2) + 4
        draw.rectangle([int(x1), label_y, int(x1) + lw + 8, label_y + lh + 4], fill=color)
        draw.text((int(x1) + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

        crop_w = int(x2 - x1)
        crop_h = int(y2 - y1)
        results_rows.append(
            f"| {i+1} | {display_name} | {det_conf*100:.0f}% | {cls_conf*100:.0f}% | {crop_w}x{crop_h} |"
        )

    results_md = f"**{len(boxes)} detection(s) found**\n\n" + "\n".join(results_rows)
    return np.array(pil_img), results_md


def _load_index():
    with open(DATA_DIR / "index.json") as f:
        return json.load(f)


INDEX = _load_index()
EXAMPLE_BARCODES = [v["barcode"] for v in INDEX["volumes"]]


def _load_volume_meta(barcode: str) -> dict:
    json_path = DATA_DIR / "volumes" / f"{barcode}.json"
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    return raw.get("metadata", {})


def get_volume_info(barcode: str) -> dict:
    json_path = DATA_DIR / "volumes" / f"{barcode}.json"
    if not json_path.exists():
        return {"barcode": barcode, "error": f"Volume '{barcode}' not found"}

    with open(json_path) as f:
        raw = json.load(f)

    detections = raw["detections"]
    classifications = {}
    captions = {}
    hashes = {}

    det_metadata = {}

    for det in detections:
        det_id = det["id"]
        if det.get("pred_class"):
            classifications[det_id] = {
                "pred_class": det["pred_class"],
                "pred_conf": det.get("classification_conf", 0),
            }
        if det.get("caption_text"):
            captions[det_id] = {
                "text": det["caption_text"],
                "lang": det.get("caption_lang", ""),
                "lang_detected": det.get("caption_lang_detected"),
                "linear_prob": det.get("caption_linear_prob"),
            }
        if det.get("image_hash"):
            hashes[det_id] = {"image_hash": det["image_hash"]}
        det_metadata[det_id] = {
            "crop_width": det.get("crop_width"),
            "crop_height": det.get("crop_height"),
            "embedding": det.get("embedding"),
            "bbox_xyxy": det.get("bbox_xyxy"),
            "image_hash": det.get("image_hash"),
        }

    return {
        "barcode": barcode,
        "volume": {
            "barcode": barcode,
            "pull_date": raw.get("pull_date"),
            "metadata": raw.get("metadata", {}),
        },
        "detections": [
            {
                "id": d["id"],
                "scan_filename": d["scan_filename"],
                "bbox_xyxy": d["bbox_xyxy"],
                "bbox_conf": d["bbox_conf"],
            }
            for d in detections
        ],
        "classifications": classifications,
        "hashes": hashes,
        "captions": captions,
        "det_metadata": det_metadata,
        "error": None,
    }


def load_volume_images(barcode: str) -> dict:
    import os
    import tempfile
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    json_path = DATA_DIR / "volumes" / f"{barcode}.json"
    with open(json_path) as f:
        raw = json.load(f)

    api = HfApi(token=token)
    cache_dir = Path(tempfile.gettempdir()) / "ve_images" / barcode
    cache_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = []
    scan_fn_map = {}
    for scan_fn in raw.get("scan_filenames", []):
        stem = Path(scan_fn).stem
        local_path = cache_dir / f"{stem}.jpg"
        if not local_path.exists():
            files_to_download.append((f"images/{barcode}/{stem}.jpg", str(local_path)))
        scan_fn_map[scan_fn] = local_path

    if files_to_download:
        api.download_bucket_files(
            bucket_id=DATA_REPO,
            files=files_to_download,
            token=token,
        )

    images = {}
    for scan_fn, local_path in scan_fn_map.items():
        if local_path.exists():
            images[scan_fn] = local_path.read_bytes()

    return images


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)


def crop_detection(image: np.ndarray, bbox_xyxy: list) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h, w = image.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return image[y1:y2, x1:x2]


def _get_font(size=24):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_annotated_page(image_np, detections_on_page, data):
    pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    img_w, img_h = pil_img.size
    font_size = max(18, min(40, img_w // 50))
    line_width = max(3, min(8, img_w // 400))
    font = _get_font(font_size)

    for det in detections_on_page:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]

        cls = data["classifications"].get(det["id"])
        class_name = (
            CLASSIFICATION_CLASS_DICT.get(cls["pred_class"], cls["pred_class"])
            if cls
            else "Unknown"
        )
        det_conf = det["bbox_conf"]
        color = CLASS_COLORS.get(class_name, (149, 165, 166))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{int(det_conf * 100)}% {class_name}"
        lbox = draw.textbbox((0, 0), label, font=font)
        lw, lh = lbox[2] - lbox[0], lbox[3] - lbox[1]

        label_y = y1 - lh - 6
        if label_y < 0:
            label_y = y2 + 4

        draw.rectangle([x1, label_y, x1 + lw + 8, label_y + lh + 4], fill=color)
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

    return np.array(pil_img)


def _group_detections_by_page(data):
    pages = defaultdict(list)
    page_order = []
    for det in data["detections"]:
        fn = det["scan_filename"]
        if fn not in pages:
            page_order.append(fn)
        pages[fn].append(det)
    return page_order, dict(pages)


def _compute_volume_stats(data):
    total = len(data["detections"])

    class_counts = defaultdict(int)
    for cls in data["classifications"].values():
        class_name = CLASSIFICATION_CLASS_DICT.get(cls["pred_class"], cls["pred_class"])
        class_counts[class_name] += 1

    captionable = 0
    failed_captions = 0
    for det in data["detections"]:
        cls = data["classifications"].get(det["id"])
        if cls and cls["pred_class"] not in CAPTION_CLASSES_EXCLUDED:
            captionable += 1
            if det["id"] not in data["captions"]:
                failed_captions += 1

    page_set = set(d["scan_filename"] for d in data["detections"])

    return {
        "total_detections": total,
        "class_counts": dict(class_counts),
        "captionable": captionable,
        "failed_captions": failed_captions,
        "page_count": len(page_set),
        "total_captions": len(data["captions"]),
    }


def format_datetime(dt):
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")


STATIC_DIR = Path(__file__).parent / "static"

_CSS = (STATIC_DIR / "viewer.css").read_text()
_JS = (STATIC_DIR / "viewer.js").read_text()


def create_gui():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#FFFFFF",
        block_background_fill="#FFFFFF",
        block_border_width="0px",
        input_background_fill="#F7F7F5",
        input_border_width="1px",
        input_border_color="#E0E0E0",
        button_primary_background_fill="#1A1A1A",
        button_primary_text_color="#FFFFFF",
        button_secondary_background_fill="transparent",
        button_secondary_border_color="#E0E0E0",
        button_secondary_text_color="#1A1A1A",
    )

    logo_path = DATA_DIR / "logo.png"
    logo_b64 = ""
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()

    with gr.Blocks(title="Visual Elements Viewer", css=_CSS, js=_JS, theme=theme) as app:
        barcode_state = gr.State("")
        page_list_state = gr.State([])
        page_idx_state = gr.State(0)

        # ── Landing Page ──────────────────────────────────────────

        with gr.Column(visible=True, elem_classes=["landing"]) as landing_page:
            if logo_b64:
                header_html = (STATIC_DIR / "landing_header.html").read_text().replace("{logo_b64}", logo_b64)
                gr.HTML(header_html)
            else:
                gr.Markdown("# Visual Elements Viewer")
            gr.HTML((STATIC_DIR / "landing_intro.html").read_text())

            status_text = gr.Markdown("")
            barcode_input = gr.Textbox(visible=False)

            spinner_html = (STATIC_DIR / "loading_spinner.html").read_text().replace("{logo_b64}", logo_b64)
            loading_spinner = gr.HTML(spinner_html, visible=False)

            with gr.Column(visible=True) as landing_content:
                gr.Markdown("### Volumes")
                volume_btns = []
                with gr.Row(elem_classes=["volume-btn"], equal_height=True):
                    for bc in EXAMPLE_BARCODES:
                        vol_meta = _load_volume_meta(bc)
                        author = vol_meta.get("author_src", "").strip() or "Unknown"
                        if len(author) > 40:
                            author = author[:37] + "..."
                        date = vol_meta.get("date1_src", "") or "n.d."
                        label = f"{bc}\n{author}, {date}"
                        btn = gr.Button(label, variant="secondary", size="sm")
                        volume_btns.append((btn, bc))

                # ── Try the Models section ────────────────────────────
                gr.HTML('<hr class="section-divider">')
                gr.Markdown(
                    "### Try the Models\n"
                    "Upload a scanned book page to run the detection "
                    "and classification pipeline. The system will identify visual elements "
                    "and classify them into categories.\n\n"
                    "*These models were trained on digitized book scans and are "
                    "unlikely to perform well on other types of images.*"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        upload_image = gr.Image(
                            label="Upload a page scan",
                            type="pil",
                            sources=["upload"],
                            show_share_button=False,
                            height=400,
                        )
                    with gr.Column(scale=1, elem_classes=["try-models-results"]):
                        output_image = gr.Image(
                            label="Results",
                            type="numpy",
                            show_share_button=False,
                            height=400,
                        )

                output_results = gr.Markdown("")

            upload_image.change(
                run_inference,
                inputs=[upload_image],
                outputs=[output_image, output_results],
            )

        # ── Volume View ───────────────────────────────────────────

        with gr.Column(visible=False) as volume_view:
            # Thumbnail navigation strip
            thumbnail_gallery = gr.Gallery(
                label="Pages",
                columns=20,
                rows=1,
                height="auto",
                object_fit="cover",
                show_label=False,
                allow_preview=False,
                elem_id="thumb-strip",
            )

            # Header: back button + volume info
            with gr.Row():
                back_btn = gr.Button(
                    "Back", variant="secondary", size="sm", scale=0, min_width=80
                )
                vol_title_md = gr.HTML("", elem_classes=["vol-header"])

            # Hidden components (kept for output wiring compatibility)
            vol_dialog_html = gr.HTML("", visible=False)
            stats_md = gr.Markdown("", visible=False)

            # Hidden elements for state/event wiring
            page_dropdown = gr.Dropdown(
                label="Page", choices=[], visible=False, interactive=True
            )
            page_counter_md = gr.Markdown("", visible=False)

            # Main content: page scan + detection details sidebar
            with gr.Row():
                with gr.Column(scale=6):
                    page_image = gr.Image(
                        label="Page Scan",
                        type="numpy",
                        elem_classes=["page-scan"],
                        show_label=False,
                        show_share_button=False,
                        show_download_button=False,
                        show_fullscreen_button=False,
                    )

                with gr.Column(scale=3, min_width=280, elem_id="detection-sidebar"):
                    crop_detail_md = gr.Markdown("")

            # Hidden gallery (kept for compatibility but not displayed)
            crop_gallery = gr.Gallery(visible=False, label="Crops")
            detection_table_md = gr.Markdown("", visible=False)

        # ── Internal helpers ──────────────────────────────────────

        def _build_detection_table(data, page_dets):
            """Build a markdown table of detection-level metadata for the current page."""
            if not page_dets:
                return ""

            rows = []
            rows.append("| ID | Class | Det % | Cls % | Size | pHash | Linear Prob | Embedding |")
            rows.append("|---|---|---|---|---|---|---|---|")

            for det in page_dets:
                det_id = det["id"]
                cls = data["classifications"].get(det_id)
                meta = data.get("det_metadata", {}).get(det_id, {})
                cap = data["captions"].get(det_id)

                class_name = (
                    CLASSIFICATION_CLASS_DICT.get(cls["pred_class"], cls["pred_class"])
                    if cls else "?"
                )
                det_conf = f"{det['bbox_conf'] * 100:.0f}%" if det["bbox_conf"] else "—"
                cls_conf = f"{cls['pred_conf'] * 100:.0f}%" if cls else "—"

                w = meta.get("crop_width")
                h = meta.get("crop_height")
                size_str = f"{w}x{h}" if w and h else "—"

                phash = meta.get("image_hash") or "—"
                if len(phash) > 12:
                    phash = phash[:12] + "..."

                linear_prob = cap.get("linear_prob") if cap else None
                lp_str = f"{linear_prob:.4f}" if linear_prob is not None else "—"

                emb = meta.get("embedding")
                if emb:
                    preview = ", ".join(f"{v:.3f}" for v in emb[:5])
                    emb_str = f"[{preview}, ...] ({len(emb)}d)"
                else:
                    emb_str = "—"

                rows.append(
                    f"| {det_id} | {class_name} | {det_conf} | {cls_conf} "
                    f"| {size_str} | {phash} | {lp_str} | {emb_str} |"
                )

            return "\n".join(rows)

        def _build_thumbnails(barcode, page_list):
            """Build small thumbnail gallery items for all pages in the volume."""
            if barcode not in _images_cache:
                _images_cache[barcode] = load_volume_images(barcode)
            images = _images_cache.get(barcode, {})

            thumbnails = []
            for i, scan_fn in enumerate(page_list):
                if scan_fn not in images:
                    continue
                img = decode_image(images[scan_fn])
                h, w = img.shape[:2]
                thumb_h = 50
                scale = thumb_h / h
                thumb = cv2.resize(img, (max(1, int(w * scale)), thumb_h), interpolation=cv2.INTER_AREA)
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                thumbnails.append((thumb_rgb, f"{i+1}"))
            return thumbnails

        def _render_page(barcode, page_list, page_idx):
            if (
                not barcode
                or not page_list
                or page_idx < 0
                or page_idx >= len(page_list)
            ):
                return None, [], "", "No pages", ""

            data = _volume_cache.get(barcode)
            if not data:
                return None, [], "", "", ""

            if barcode not in _images_cache:
                _images_cache[barcode] = load_volume_images(barcode)

            images = _images_cache.get(barcode, {})
            scan_filename = page_list[page_idx]
            page_dets = [
                d for d in data["detections"] if d["scan_filename"] == scan_filename
            ]

            counter_text = (
                f"**{page_idx + 1}/{len(page_list)}** "
                f"{scan_filename} ({len(page_dets)} det)"
            )

            if scan_filename not in images:
                return None, [], "", counter_text, ""

            full_image = decode_image(images[scan_filename])
            annotated = draw_annotated_page(full_image, page_dets, data)

            max_dim = 1400
            h, w = annotated.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                annotated = cv2.resize(
                    annotated, (int(w * scale), int(h * scale))
                )

            gallery_items = []
            detail_parts = []
            detail_parts.append(f"**Page {page_idx + 1}/{len(page_list)}** — {len(page_dets)} detection(s)\n")

            for det in page_dets:
                crop = crop_detection(full_image, det["bbox_xyxy"])
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                cls = data["classifications"].get(det["id"])
                class_name = (
                    CLASSIFICATION_CLASS_DICT.get(
                        cls["pred_class"], cls["pred_class"]
                    )
                    if cls
                    else "?"
                )
                conf_pct = f"{det['bbox_conf'] * 100:.0f}%"
                gallery_items.append(
                    (crop_rgb, f"#{det['id']} {class_name} ({conf_pct})")
                )

                det_id = det["id"]
                meta = data.get("det_metadata", {}).get(det_id, {})
                cls_conf = f"{cls['pred_conf'] * 100:.0f}%" if cls else "?"

                part = f"#### #{det_id} {class_name}\n"
                part += f"Det: {conf_pct} | Cls: {cls_conf}"

                crop_w = meta.get("crop_width")
                crop_h = meta.get("crop_height")
                if crop_w and crop_h:
                    part += f" | {crop_w}x{crop_h}"

                phash = meta.get("image_hash")
                if phash:
                    part += f"\n\npHash: `{phash[:16]}`"

                cap = data["captions"].get(det_id)
                if cap:
                    part += f"\n\n> {cap['text']}"
                    lang = cap.get("lang", "")
                    if lang:
                        part += f"\n\n*{lang}*"
                    lp = cap.get("linear_prob")
                    if lp is not None:
                        part += f"\n\nCaption linear logprob: `{lp:.4f}`"
                else:
                    excluded = cls and cls["pred_class"] in CAPTION_CLASSES_EXCLUDED
                    if excluded:
                        part += "\n\n*No caption (excluded class)*"
                    else:
                        part += "\n\n*Caption missing*"

                detail_parts.append(part)

            details_md = "\n\n---\n\n".join(detail_parts) if detail_parts else ""

            return annotated, gallery_items, details_md, counter_text, ""

        def _empty_load_result(message=""):
            return [
                gr.update(),
                gr.update(),
                gr.update(visible=False),
                gr.update(visible=True),
                message,
                "",
                [],
                0,
                "",
                "",
                "",
                gr.update(choices=[]),
                "",
                None,
                [],
                "",
                "",
                [],
            ]

        def handle_load(barcode):
            if not barcode or not barcode.strip():
                return _empty_load_result("Enter a barcode to begin.")

            barcode = barcode.strip()
            data = get_volume_info(barcode)

            if data.get("error"):
                return _empty_load_result(f"**Error:** {data['error']}")

            _volume_cache[barcode] = data
            _images_cache.pop(barcode, None)

            stats = _compute_volume_stats(data)
            page_order, _ = _group_detections_by_page(data)

            vol = data["volume"]
            meta = vol.get("metadata", {})
            title = meta.get("title_src", "Untitled")
            author = meta.get("author_src", "").strip() or "Unknown"
            date = meta.get("date1_src", "") or "n.d."
            topic = meta.get("topic_or_subject_gen") or meta.get("topic_or_subject_src") or ""

            SKIP_META_KEYS = {
                "text_analysis_gen",
                "likely_duplicates_barcodes_gen",
                "identifiers_src",
                "language_distribution_gen",
                "topic_or_subject_score_gen",
                "topic_or_subject_src",
                "hathitrust_data_ext",
            }

            denom = stats["total_detections"] or 1
            class_parts = []
            for class_name, count in sorted(
                stats["class_counts"].items(), key=lambda x: -x[1]
            ):
                pct = count / denom * 100
                class_parts.append(f"{class_name}: {count} ({pct:.0f}%)")
            classes_str = " &middot; ".join(class_parts)

            vol_info_rows = f"<tr><th>Barcode</th><td><code style='font-family:monospace'>{vol['barcode']}</code></td></tr>"
            vol_info_rows += f"<tr><th>Pull Date</th><td>{format_datetime(vol.get('pull_date'))}</td></tr>"
            vol_info_rows += f"<tr><th>Detections</th><td>{stats['total_detections']}</td></tr>"
            vol_info_rows += f"<tr><th>Pages</th><td>{stats['page_count']}</td></tr>"
            vol_info_rows += f"<tr><th>Captions</th><td>{stats['total_captions']}/{stats['captionable']}</td></tr>"
            if stats["failed_captions"]:
                vol_info_rows += f"<tr><th>Failed Captions</th><td>{stats['failed_captions']}</td></tr>"
            vol_info_rows += f"<tr><th>Classes</th><td>{classes_str}</td></tr>"
            if meta:
                for key, val in meta.items():
                    if key in SKIP_META_KEYS:
                        continue
                    val_str = str(val)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    vol_info_rows += f"<tr><th>{key}</th><td>{val_str}</td></tr>"

            subtitle_parts = [f"{author}, {date}"]
            if topic:
                subtitle_parts.append(topic)
            subtitle_parts.append(f'<code style="font-family:monospace">{barcode}</code>')
            subtitle = " · ".join(subtitle_parts)

            vol_header = (
                (STATIC_DIR / "vol_header.html").read_text()
                .replace("{title}", title)
                .replace("{subtitle}", subtitle)
                .replace("{vol_info_rows}", vol_info_rows)
            )

            page_img, gallery, details, counter, table = _render_page(barcode, page_order, 0)
            thumbnails = _build_thumbnails(barcode, page_order)

            return [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                barcode,
                page_order,
                0,
                vol_header,
                "",
                "",
                gr.update(
                    choices=page_order,
                    value=page_order[0] if page_order else None,
                ),
                counter,
                page_img,
                gallery,
                details,
                table,
                thumbnails,
            ]

        def handle_page_select(page_filename, barcode, page_list):
            if not page_filename or not page_list:
                return None, [], "", "", "", 0
            try:
                idx = page_list.index(page_filename)
            except ValueError:
                idx = 0
            page_img, gallery, details, counter, table = _render_page(barcode, page_list, idx)
            return page_img, gallery, details, counter, table, idx


        def handle_back():
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

        def handle_thumbnail_select(evt: gr.SelectData, barcode, page_list):
            idx = evt.index
            if not page_list or idx >= len(page_list):
                return None, [], "", "", "", idx
            page_img, gallery, details, counter, table = _render_page(barcode, page_list, idx)
            return page_img, gallery, details, counter, table, idx

        # ── Wire up events ────────────────────────────────────────

        load_outputs = [
            landing_page,
            volume_view,
            loading_spinner,
            landing_content,
            status_text,
            barcode_state,
            page_list_state,
            page_idx_state,
            vol_title_md,
            vol_dialog_html,
            stats_md,
            page_dropdown,
            page_counter_md,
            page_image,
            crop_gallery,
            crop_detail_md,
            detection_table_md,
            thumbnail_gallery,
        ]

        def _show_loading():
            return gr.update(visible=True), gr.update(visible=False)

        for btn, bc in volume_btns:
            btn.click(lambda x=bc: x, outputs=[barcode_input]).then(
                _show_loading, outputs=[loading_spinner, landing_content]
            ).then(
                handle_load, inputs=[barcode_input], outputs=load_outputs
            )

        back_btn.click(
            handle_back,
            outputs=[landing_page, volume_view, landing_content, loading_spinner],
        )

        thumbnail_gallery.select(
            handle_thumbnail_select,
            inputs=[barcode_state, page_list_state],
            outputs=[
                page_image,
                crop_gallery,
                crop_detail_md,
                page_counter_md,
                detection_table_md,
                page_idx_state,
            ],
        )

        page_dropdown.change(
            handle_page_select,
            inputs=[page_dropdown, barcode_state, page_list_state],
            outputs=[
                page_image,
                crop_gallery,
                crop_detail_md,
                page_counter_md,
                detection_table_md,
                page_idx_state,
            ],
        )

    return app


app = create_gui()
app.launch()
