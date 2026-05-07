"""
Pipeline Viewer GUI - Interactive visual walkthrough of pipeline stages for a volume.

Launch with: python pipeline.py export viewer-gui
"""

import click
import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from collections import defaultdict

from const import CLASSIFICATION_CLASS_DICT, CAPTION_CLASSES_EXCLUDED


CLASS_COLORS = {
    "Image/Illustration": (46, 204, 113),
    "Ex Libris/Decorative": (230, 126, 34),
    "Music": (52, 152, 219),
    "Chart/Graph": (155, 89, 182),
    "Artifact": (241, 196, 15),
    "Other": (149, 165, 166),
}

_volume_cache = {}
_images_cache = {}


def format_datetime(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_volume_info(barcode: str) -> dict:
    """Load all pipeline data for a volume."""
    from models import (
        IBVolume,
        PipelineBatchItem,
        Detection,
        Classification,
        ImageHash,
        ImageEmbedding,
        Caption,
        DedupedHash,
        DedupedEmbedding,
    )

    result = {
        "barcode": barcode,
        "volume": None,
        "batch_item": None,
        "batch": None,
        "run": None,
        "detections": [],
        "classifications": {},
        "hashes": {},
        "embeddings": {},
        "captions": {},
        "deduped_hashes": {},
        "deduped_embeddings": {},
        "error": None,
    }

    try:
        volume = IBVolume.get(IBVolume.barcode == barcode)
        result["volume"] = {
            "barcode": volume.barcode,
            "pull_date": volume.pull_date,
            "metadata": volume.metadata,
        }
    except IBVolume.DoesNotExist:
        result["error"] = f"Volume '{barcode}' not found in database"
        return result

    try:
        batch_item = (
            PipelineBatchItem.select()
            .where(PipelineBatchItem.ib_volume == barcode)
            .order_by(PipelineBatchItem.id_pipeline_batch_item.desc())
            .get()
        )
        batch = batch_item.pipeline_batch
        run = batch.pipeline_run

        result["batch_item"] = {
            "id": batch_item.id_pipeline_batch_item,
            "obj": batch_item,
        }
        result["batch"] = {
            "id": batch.id_pipeline_batch,
            "node_name": batch.node_name,
            "started_date": batch.started_date,
            "ended_date": batch.ended_date,
        }
        result["run"] = {
            "id": run.id_pipeline_run,
            "created_date": run.created_date,
            "items_total": run.items_total,
            "batches_total": run.batches_total,
        }
    except PipelineBatchItem.DoesNotExist:
        return result

    detections = list(
        Detection.select()
        .where(Detection.pipeline_batch_item == batch_item.id_pipeline_batch_item)
        .order_by(Detection.scan_filename, Detection.id_detection)
    )
    result["detections"] = [
        {
            "id": det.id_detection,
            "scan_filename": det.scan_filename,
            "bbox_xyxy": det.bbox_xyxy,
            "bbox_conf": det.bbox_conf,
        }
        for det in detections
    ]

    if not detections:
        return result

    detection_ids = [det.id_detection for det in detections]

    classifications = list(
        Classification.select().where(Classification.detection_id.in_(detection_ids))
    )
    result["classifications"] = {
        cls.detection_id: {
            "pred_class": cls.pred_class,
            "pred_conf": cls.pred_conf,
        }
        for cls in classifications
    }

    hashes = list(ImageHash.select().where(ImageHash.detection_id.in_(detection_ids)))
    result["hashes"] = {h.detection_id: {"image_hash": h.image_hash} for h in hashes}

    embeddings = list(
        ImageEmbedding.select(
            ImageEmbedding.id_embedding,
            ImageEmbedding.detection_id,
        ).where(ImageEmbedding.detection_id.in_(detection_ids))
    )
    result["embeddings"] = {e.detection_id: {"id": e.id_embedding} for e in embeddings}

    captions = list(Caption.select().where(Caption.detection_id.in_(detection_ids)))
    result["captions"] = {
        cap.detection_id: {"text": cap.text, "lang": cap.lang} for cap in captions
    }

    deduped_hashes = list(
        DedupedHash.select().where(DedupedHash.detection_id.in_(detection_ids))
    )
    result["deduped_hashes"] = {
        dh.detection_id: {"group_id": dh.group_id} for dh in deduped_hashes
    }

    deduped_embeddings = list(
        DedupedEmbedding.select().where(DedupedEmbedding.detection_id.in_(detection_ids))
    )
    result["deduped_embeddings"] = {
        de.detection_id: {"group_id": de.group_id} for de in deduped_embeddings
    }

    return result


def load_volume_images(barcode: str) -> dict:
    """Load volume images from cache/S3."""
    from models import PipelineBatchItem

    try:
        batch_item = (
            PipelineBatchItem.select()
            .where(PipelineBatchItem.ib_volume == barcode)
            .order_by(PipelineBatchItem.id_pipeline_batch_item.desc())
            .get()
        )
        item_data = batch_item.get_data()
        return item_data.images
    except Exception as e:
        logger.error(f"Failed to load volume images: {e}")
        return {}


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array (BGR)."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)


def crop_detection(image: np.ndarray, bbox_xyxy: list) -> np.ndarray:
    """Crop a detection from an image."""
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


def draw_annotated_page(
    image_np: np.ndarray, detections_on_page: list, data: dict
) -> np.ndarray:
    """Draw bounding boxes with confidence labels using Pillow's ImageDraw."""
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
    """Group detections by scan_filename. Returns (page_order, pages_dict)."""
    pages = defaultdict(list)
    page_order = []
    for det in data["detections"]:
        fn = det["scan_filename"]
        if fn not in pages:
            page_order.append(fn)
        pages[fn].append(det)
    return page_order, dict(pages)


def _compute_volume_stats(data):
    """Compute per-volume statistics."""
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
        "deduped_hash_groups": len(
            set(v["group_id"] for v in data["deduped_hashes"].values())
        ),
        "deduped_embedding_groups": len(
            set(v["group_id"] for v in data["deduped_embeddings"].values())
        ),
    }


EXAMPLE_BARCODES = [
    "32044004431664",
    "32044004479606",
    "32044004462776",
    "32044004465357",
    "32044004458956",
    "32044004471777",
    "32044004460424",
    "32044004477063",
]


_CSS = """
.page-scan img {
    max-height: 80vh !important;
    object-fit: contain !important;
}
.landing {
    max-width: 900px;
    margin: 40px auto;
}
"""


def create_gui():
    """Create the Gradio GUI interface."""

    with gr.Blocks(title="Pipeline Viewer") as app:
        barcode_state = gr.State("")
        page_list_state = gr.State([])
        page_idx_state = gr.State(0)

        # ── Landing Page ──────────────────────────────────────────

        with gr.Column(visible=True, elem_classes=["landing"]) as landing_page:
            gr.Markdown(
                "# Pipeline Viewer\n"
                "Interactive walkthrough of pipeline stages for a volume. "
                "Select an example or enter a barcode."
            )

            with gr.Row():
                barcode_input = gr.Textbox(
                    label="Volume Barcode",
                    placeholder="e.g. 32044004479606",
                    scale=3,
                )
                load_btn = gr.Button("Load Volume", variant="primary", scale=1)

            status_text = gr.Markdown("")

            gr.Markdown("### Example Volumes")
            example_btns = []
            for i in range(0, len(EXAMPLE_BARCODES), 4):
                with gr.Row():
                    for bc in EXAMPLE_BARCODES[i : i + 4]:
                        btn = gr.Button(bc, variant="secondary", size="sm")
                        example_btns.append((btn, bc))

        # ── Volume View ───────────────────────────────────────────

        with gr.Column(visible=False) as volume_view:
            # Row 1: back, title, navigation mode, filters
            with gr.Row():
                back_btn = gr.Button(
                    "Back", variant="secondary", size="sm", scale=0, min_width=80
                )
                vol_title_md = gr.Markdown("")
                nav_mode = gr.Radio(
                    ["By Page", "By Class", "By Confidence"],
                    value="By Page",
                    label="Navigation",
                    scale=3,
                )
                class_filter = gr.Dropdown(
                    label="Filter Class",
                    choices=[],
                    visible=False,
                    scale=2,
                    interactive=True,
                )
                conf_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=30,
                    step=5,
                    label="Min Confidence %",
                    visible=False,
                    scale=2,
                )

            # Row 2: page navigation
            with gr.Row():
                prev_page_btn = gr.Button(
                    "Prev", size="sm", scale=0, min_width=80
                )
                page_dropdown = gr.Dropdown(
                    label="Page", choices=[], scale=4, interactive=True
                )
                next_page_btn = gr.Button(
                    "Next", size="sm", scale=0, min_width=80
                )
                page_counter_md = gr.Markdown("")

            # Three-column layout
            with gr.Row():
                # Left sidebar: volume info + stats
                with gr.Column(scale=2, min_width=180):
                    gr.Markdown("#### Volume Info")
                    volume_info_md = gr.Markdown(
                        "", max_height="35vh", container=True
                    )
                    gr.Markdown("#### Statistics")
                    stats_md = gr.Markdown(
                        "", max_height="35vh", container=True
                    )

                # Center: page scan with overlaid bounding boxes
                with gr.Column(scale=5):
                    page_image = gr.Image(
                        label="Page Scan",
                        type="numpy",
                        elem_classes=["page-scan"],
                    )

                # Right sidebar: crop gallery + detail
                with gr.Column(scale=3, min_width=200):
                    gr.Markdown("#### Detections on Page")
                    crop_gallery = gr.Gallery(
                        label="Crops (click for details)",
                        columns=2,
                        height=200,
                        object_fit="contain",
                    )
                    crop_detail_md = gr.Markdown(
                        "", max_height="40vh", container=True
                    )

        # ── Internal helpers ──────────────────────────────────────

        def _render_page(barcode, page_list, page_idx):
            """Render annotated page scan and crop gallery for given page."""
            if (
                not barcode
                or not page_list
                or page_idx < 0
                or page_idx >= len(page_list)
            ):
                return None, [], "", "No pages"

            data = _volume_cache.get(barcode)
            if not data:
                return None, [], "", ""

            if barcode not in _images_cache:
                logger.info(f"Loading images for {barcode}...")
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
                return None, [], "", counter_text

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
            for det in page_dets:
                crop = crop_detection(full_image, det["bbox_xyxy"])
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

                cls_conf = f"{cls['pred_conf'] * 100:.0f}%" if cls else "?"
                part = f"**#{det['id']}** {class_name} ({cls_conf})"
                cap = data["captions"].get(det["id"])
                if cap:
                    text = cap["text"]
                    if len(text) > 80:
                        text = text[:77] + "..."
                    part += f"\n> {text}\n*{cap['lang']}*"
                else:
                    excluded = cls and cls["pred_class"] in CAPTION_CLASSES_EXCLUDED
                    if excluded:
                        part += "\n*No caption (excluded)*"
                    else:
                        part += "\n*Caption missing*"
                detail_parts.append(part)

            details_md = "\n\n---\n\n".join(detail_parts) if detail_parts else ""

            return annotated, gallery_items, details_md, counter_text

        def _filter_page_list(barcode, mode, class_val, conf_val):
            """Get filtered/sorted page list based on navigation mode."""
            data = _volume_cache.get(barcode)
            if not data:
                return []

            page_order, pages = _group_detections_by_page(data)

            if mode == "By Class" and class_val:
                filtered = []
                for fn in page_order:
                    for det in pages[fn]:
                        cls = data["classifications"].get(det["id"])
                        if cls:
                            cn = CLASSIFICATION_CLASS_DICT.get(
                                cls["pred_class"], cls["pred_class"]
                            )
                            if cn == class_val:
                                filtered.append(fn)
                                break
                return filtered

            if mode == "By Confidence":
                threshold = (conf_val or 30) / 100.0

                def max_conf(fn):
                    return max(
                        (d["bbox_conf"] for d in pages[fn]), default=0
                    )

                filtered = [fn for fn in page_order if max_conf(fn) >= threshold]
                filtered.sort(key=max_conf, reverse=True)
                return filtered

            return page_order

        def _empty_load_result(message=""):
            """Return tuple for load_volume when there's nothing to show."""
            return [
                gr.update(),  # landing_page
                gr.update(),  # volume_view
                message,  # status_text
                "",  # barcode_state
                [],  # page_list_state
                0,  # page_idx_state
                "",  # vol_title_md
                "",  # volume_info_md
                "",  # stats_md
                gr.update(choices=[]),  # page_dropdown
                "",  # page_counter_md
                None,  # page_image
                [],  # crop_gallery
                "",  # crop_detail_md
                gr.update(choices=[], visible=False),  # class_filter
                gr.update(visible=False, value=30),  # conf_slider
                gr.update(value="By Page"),  # nav_mode
            ]

        # ── Event handlers ────────────────────────────────────────

        def handle_load(barcode):
            """Load a volume and switch to volume view."""
            if not barcode or not barcode.strip():
                return _empty_load_result("Enter a barcode to begin.")

            barcode = barcode.strip()
            data = get_volume_info(barcode)

            if data["error"]:
                return _empty_load_result(f"**Error:** {data['error']}")

            _volume_cache[barcode] = data
            _images_cache.pop(barcode, None)

            stats = _compute_volume_stats(data)
            page_order, _ = _group_detections_by_page(data)

            # Build volume info markdown
            vol = data["volume"]
            vol_info = f"**Barcode:** {vol['barcode']}\n\n"
            vol_info += f"**Pull Date:** {format_datetime(vol['pull_date'])}\n\n"
            if vol["metadata"]:
                for key, val in vol["metadata"].items():
                    val_str = str(val)
                    if len(val_str) > 60:
                        val_str = val_str[:57] + "..."
                    vol_info += f"**{key}:** {val_str}\n\n"
            if data["batch"]:
                batch = data["batch"]
                vol_info += "---\n\n"
                vol_info += f"**Batch:** {batch['id']}\n\n"
                vol_info += f"**Node:** {batch['node_name']}\n\n"
                vol_info += f"**Processed:** {format_datetime(batch['started_date'])}\n"

            # Build stats markdown
            stats_text = f"**Detections:** {stats['total_detections']}\n\n"
            stats_text += f"**Pages:** {stats['page_count']}\n\n"
            stats_text += "---\n\n**Classes:**\n\n"
            denom = stats["total_detections"] or 1
            for class_name, count in sorted(
                stats["class_counts"].items(), key=lambda x: -x[1]
            ):
                pct = count / denom * 100
                stats_text += f"- {class_name}: {count} ({pct:.0f}%)\n"

            stats_text += "\n---\n\n"
            stats_text += f"**Captions:** {stats['total_captions']}/{stats['captionable']}\n\n"
            if stats["failed_captions"]:
                stats_text += f"**Failed:** {stats['failed_captions']}\n\n"
            stats_text += f"**Dedupe H:** {stats['deduped_hash_groups']} grp\n\n"
            stats_text += f"**Dedupe E:** {stats['deduped_embedding_groups']} grp\n"

            # Unique classes for filter dropdown
            classes = sorted(
                set(
                    CLASSIFICATION_CLASS_DICT.get(
                        c["pred_class"], c["pred_class"]
                    )
                    for c in data["classifications"].values()
                )
            )

            # Render first page
            page_img, gallery, details, counter = _render_page(barcode, page_order, 0)

            return [
                gr.update(visible=False),  # landing_page
                gr.update(visible=True),  # volume_view
                "",  # status_text
                barcode,  # barcode_state
                page_order,  # page_list_state
                0,  # page_idx_state
                f"### Volume {barcode}",  # vol_title_md
                vol_info,  # volume_info_md
                stats_text,  # stats_md
                gr.update(
                    choices=page_order,
                    value=page_order[0] if page_order else None,
                ),  # page_dropdown
                counter,  # page_counter_md
                page_img,  # page_image
                gallery,  # crop_gallery
                details,  # crop_detail_md
                gr.update(choices=classes, visible=False, value=None),  # class_filter
                gr.update(visible=False, value=30),  # conf_slider
                gr.update(value="By Page"),  # nav_mode
            ]

        def handle_page_select(page_filename, barcode, page_list):
            """Handle page selection from dropdown."""
            if not page_filename or not page_list:
                return None, [], "", "", 0
            try:
                idx = page_list.index(page_filename)
            except ValueError:
                idx = 0
            page_img, gallery, details, counter = _render_page(barcode, page_list, idx)
            return page_img, gallery, details, counter, idx

        def handle_prev_page(barcode, page_list, page_idx):
            """Navigate to previous page."""
            if not page_list:
                return None, [], "", "", gr.update(), 0
            new_idx = (page_idx - 1) % len(page_list)
            page_img, gallery, details, counter = _render_page(
                barcode, page_list, new_idx
            )
            return (
                page_img,
                gallery,
                details,
                counter,
                gr.update(value=page_list[new_idx]),
                new_idx,
            )

        def handle_next_page(barcode, page_list, page_idx):
            """Navigate to next page."""
            if not page_list:
                return None, [], "", "", gr.update(), 0
            new_idx = (page_idx + 1) % len(page_list)
            page_img, gallery, details, counter = _render_page(
                barcode, page_list, new_idx
            )
            return (
                page_img,
                gallery,
                details,
                counter,
                gr.update(value=page_list[new_idx]),
                new_idx,
            )

        def handle_nav_mode_change(mode):
            """Toggle visibility of filter controls based on navigation mode."""
            return (
                gr.update(visible=mode == "By Class"),
                gr.update(visible=mode == "By Confidence"),
            )

        def handle_apply_filter(barcode, mode, class_val, conf_val):
            """Recompute page list based on current filter and render first page."""
            page_list = _filter_page_list(barcode, mode, class_val, conf_val)
            if not page_list:
                return (
                    [],
                    0,
                    gr.update(choices=[], value=None),
                    None,
                    [],
                    "",
                    "No pages match filter",
                )
            page_img, gallery, details, counter = _render_page(
                barcode, page_list, 0
            )
            return (
                page_list,
                0,
                gr.update(choices=page_list, value=page_list[0]),
                page_img,
                gallery,
                details,
                counter,
            )

        def handle_crop_select(
            evt: gr.SelectData, barcode, page_list, page_idx
        ):
            """Show details for a selected crop in the gallery."""
            data = _volume_cache.get(barcode)
            if not data or not page_list or page_idx >= len(page_list):
                return ""

            scan_filename = page_list[page_idx]
            page_dets = [
                d
                for d in data["detections"]
                if d["scan_filename"] == scan_filename
            ]

            if evt.index >= len(page_dets):
                return ""

            det = page_dets[evt.index]
            bbox = det["bbox_xyxy"]

            cls = data["classifications"].get(det["id"])
            class_name = (
                CLASSIFICATION_CLASS_DICT.get(cls["pred_class"], cls["pred_class"])
                if cls
                else "?"
            )
            cls_conf = f"{cls['pred_conf'] * 100:.0f}%" if cls else "?"

            info = f"### #{det['id']} {class_name}\n\n"
            info += f"**Det:** {det['bbox_conf'] * 100:.0f}%"
            info += f" | **Cls:** {cls_conf}\n\n"

            cap = data["captions"].get(det["id"])
            if cap:
                info += f"> {cap['text']}\n\n"
                info += f"*{cap['lang']}*\n\n"
            else:
                excluded = cls and cls["pred_class"] in CAPTION_CLASSES_EXCLUDED
                if excluded:
                    info += "*No caption (excluded class)*\n\n"
                else:
                    info += "*Caption missing*\n\n"

            dh = data["deduped_hashes"].get(det["id"])
            de = data["deduped_embeddings"].get(det["id"])
            if dh or de:
                parts = []
                if dh:
                    parts.append(f"H:{dh['group_id']}")
                if de:
                    parts.append(f"E:{de['group_id']}")
                info += f"Dedupe: {' / '.join(parts)}\n"

            return info

        def handle_back():
            """Return to landing page."""
            return gr.update(visible=True), gr.update(visible=False)

        # ── Wire up events ────────────────────────────────────────

        load_outputs = [
            landing_page,
            volume_view,
            status_text,
            barcode_state,
            page_list_state,
            page_idx_state,
            vol_title_md,
            volume_info_md,
            stats_md,
            page_dropdown,
            page_counter_md,
            page_image,
            crop_gallery,
            crop_detail_md,
            class_filter,
            conf_slider,
            nav_mode,
        ]

        load_btn.click(handle_load, inputs=[barcode_input], outputs=load_outputs)
        barcode_input.submit(
            handle_load, inputs=[barcode_input], outputs=load_outputs
        )

        for btn, bc in example_btns:
            btn.click(lambda x=bc: x, outputs=[barcode_input]).then(
                handle_load, inputs=[barcode_input], outputs=load_outputs
            )

        back_btn.click(
            handle_back, outputs=[landing_page, volume_view]
        )

        page_nav_outputs = [
            page_image,
            crop_gallery,
            crop_detail_md,
            page_counter_md,
            page_dropdown,
            page_idx_state,
        ]

        prev_page_btn.click(
            handle_prev_page,
            inputs=[barcode_state, page_list_state, page_idx_state],
            outputs=page_nav_outputs,
        )
        next_page_btn.click(
            handle_next_page,
            inputs=[barcode_state, page_list_state, page_idx_state],
            outputs=page_nav_outputs,
        )

        page_dropdown.change(
            handle_page_select,
            inputs=[page_dropdown, barcode_state, page_list_state],
            outputs=[
                page_image,
                crop_gallery,
                crop_detail_md,
                page_counter_md,
                page_idx_state,
            ],
        )

        crop_gallery.select(
            handle_crop_select,
            inputs=[barcode_state, page_list_state, page_idx_state],
            outputs=[crop_detail_md],
        )

        filter_outputs = [
            page_list_state,
            page_idx_state,
            page_dropdown,
            page_image,
            crop_gallery,
            crop_detail_md,
            page_counter_md,
        ]

        nav_mode.change(
            handle_nav_mode_change,
            inputs=[nav_mode],
            outputs=[class_filter, conf_slider],
        ).then(
            handle_apply_filter,
            inputs=[barcode_state, nav_mode, class_filter, conf_slider],
            outputs=filter_outputs,
        )

        class_filter.change(
            handle_apply_filter,
            inputs=[barcode_state, nav_mode, class_filter, conf_slider],
            outputs=filter_outputs,
        )

        conf_slider.release(
            handle_apply_filter,
            inputs=[barcode_state, nav_mode, class_filter, conf_slider],
            outputs=filter_outputs,
        )

    return app


@click.command("viewer-gui")
@click.option("--port", type=int, default=7860, help="Port to run the GUI on")
@click.option("--share", is_flag=True, help="Create a public shareable link")
def viewer_gui(port, share):
    """
    Launch the interactive Pipeline Viewer GUI.

    Opens a web browser with a visual interface for exploring
    pipeline stages for any volume.
    """
    logger.info(f"Starting Pipeline Viewer GUI on port {port}...")
    app = create_gui()
    app.launch(server_port=port, share=share, css=_CSS, theme=gr.themes.Soft())
