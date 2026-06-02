"""
Export a self-contained HuggingFace Space with static data from the pipeline.

Generates a Gradio app + pre-exported volume data (images + JSON metadata)
ready to push to a HF Space repo.
"""

import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import cv2
import numpy as np
from loguru import logger

from utils import get_db
from const import CLASSIFICATION_CLASS_DICT, CAPTION_CLASSES_EXCLUDED

HF_SPACE_REPO = "institutional/institutional-books-hl-visual-elements"
HF_DATA_REPO = "institutional/institutional-books-hl-visual-elements-demo"


def _get_raw_connection():
    import psycopg2

    db = get_db()
    conn = db.connection()
    try:
        conn.cursor().execute("SELECT 1")
        conn.rollback()
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        db.close()
        db.connect()
        conn = db.connection()
    if conn.autocommit:
        conn.autocommit = False
    return conn


VOLUME_BARCODES = ["HS1CL5", "HNZY4S", "32044094151479", "32044050659580", "HN5JD5"]


def _fetch_volume_detections(barcode: str) -> list[dict]:
    """Fetch all detections for a barcode from filtered_dataset."""
    conn = _get_raw_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT *
            FROM filtered_dataset
            WHERE barcode = %s
            ORDER BY scan_filename, id_detection
        """, (barcode,))
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description]
        cur.close()
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        conn.rollback()


def _fetch_volume_metadata(barcode: str) -> dict:
    """Fetch volume metadata from IBVolume."""
    from models import IBVolume

    try:
        volume = IBVolume.get(IBVolume.barcode == barcode)
        pull_date = volume.pull_date
        return {
            "pull_date": pull_date.isoformat() if pull_date else None,
            "metadata": volume.metadata or {},
        }
    except IBVolume.DoesNotExist:
        return {"pull_date": None, "metadata": {}}


def _load_volume_images(barcode: str) -> dict:
    """Load volume page scan images from S3."""
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
        logger.error(f"Failed to load images for {barcode}: {e}")
        return {}


def _compress_image(image_bytes: bytes, max_dim: int = 1400, quality: int = 80) -> tuple[bytes, float]:
    """Decode image, resize to max_dim, encode as JPEG. Returns (jpeg_bytes, scale_factor)."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes, 1.0

    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes(), scale


def _build_volume_json(barcode: str, vol_meta: dict, rows: list[dict], scale_factors: dict[str, float]) -> dict:
    """Build the per-volume JSON structure with bounding boxes scaled to match compressed images."""
    scan_filenames = list(dict.fromkeys(r["scan_filename"] for r in rows))

    detections = []
    for row in rows:
        bbox = list(row["bbox_xyxy"]) if row["bbox_xyxy"] else None
        if bbox:
            s = scale_factors.get(row["scan_filename"], 1.0)
            bbox = [v * s for v in bbox]

        bbox_xywh = list(row["bbox_xywh"]) if row.get("bbox_xywh") else None
        crop_width = int(round(bbox_xywh[2])) if bbox_xywh and len(bbox_xywh) >= 4 else None
        crop_height = int(round(bbox_xywh[3])) if bbox_xywh and len(bbox_xywh) >= 4 else None

        embedding = row.get("embedding")
        if embedding is not None:
            if isinstance(embedding, str):
                embedding = [float(x) for x in embedding.strip("[]").split(",")]
            else:
                embedding = [float(x) for x in embedding]

        det = {
            "id": row["id_detection"],
            "scan_filename": row["scan_filename"],
            "bbox_xyxy": bbox,
            "bbox_conf": float(row["bbox_conf"]) if row["bbox_conf"] is not None else None,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "pred_class": row["pred_class"],
            "classification_conf": float(row["classification_conf"]) if row["classification_conf"] is not None else None,
            "caption_text": row.get("caption_text"),
            "caption_lang": row.get("caption_lang"),
            "caption_lang_detected": row.get("caption_lang_detected"),
            "caption_linear_prob": float(row["caption_linear_prob"]) if row.get("caption_linear_prob") is not None else None,
            "image_hash": row.get("image_hash"),
            "embedding": embedding,
        }
        detections.append(det)

    return {
        "barcode": barcode,
        "pull_date": vol_meta.get("pull_date"),
        "metadata": vol_meta.get("metadata", {}),
        "scan_filenames": scan_filenames,
        "detections": detections,
    }


def _generate_app_py() -> str:
    """Generate the static Gradio app.py for the HF Space."""
    template_path = Path(__file__).parent / "templates" / "viewer_space_app.py"
    return template_path.read_text()


def _copy_static_assets(output_path: Path):
    """Copy the static CSS/HTML templates alongside app.py."""
    static_src = Path(__file__).parent / "templates" / "static"
    static_dst = output_path / "static"
    if static_src.exists():
        shutil.copytree(static_src, static_dst, dirs_exist_ok=True)


def _generate_requirements_txt() -> str:
    return "gradio>=5.33.0\nnumpy\nopencv-python-headless\nPillow\nultralytics\ntorch\ntorchvision\n"


def _generate_readme() -> str:
    return """---
title: Institutional Books Visual Elements Viewer
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.33.0"
app_file: app.py
pinned: false
---

# Institutional Books Visual Elements Viewer

Interactive viewer for the visual elements detection pipeline.
Browse pre-processed volumes with bounding box annotations, classifications, and captions.
"""


@click.command("viewer-space")
@click.option("--output-dir", type=click.Path(), default="./space_output", help="Output directory for the Space")
@click.option("--detections-only", is_flag=True, help="Only export pages that have detections")
@click.option("--push", is_flag=True, help="Push to HuggingFace Space after export")
def viewer_space(output_dir, detections_only, push):
    """
    Export a static HuggingFace Space with pre-baked volume data.

    Generates a complete Gradio app directory with images and metadata
    for the volumes defined in VOLUME_BARCODES.
    """
    output_path = Path(output_dir)

    if output_path.exists():
        logger.warning(f"Output directory {output_path} exists, clearing it")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)
    (output_path / "data" / "volumes").mkdir(parents=True)
    (output_path / "data" / "images").mkdir(parents=True)

    barcodes = VOLUME_BARCODES
    logger.info(f"Exporting {len(barcodes)} volumes: {barcodes}")

    def _process_volume(barcode: str) -> dict:
        """Process a single volume: fetch data, download images, compress, write to disk."""
        logger.info(f"  [{barcode}] Fetching detections...")
        rows = _fetch_volume_detections(barcode)
        logger.info(f"  [{barcode}] {len(rows)} detections")

        vol_meta = _fetch_volume_metadata(barcode)

        logger.info(f"  [{barcode}] Loading page scans from S3...")
        images = _load_volume_images(barcode)
        logger.info(f"  [{barcode}] {len(images)} pages loaded")

        pages_with_detections = set(r["scan_filename"] for r in rows)

        img_dir = output_path / "data" / "images" / barcode
        img_dir.mkdir(parents=True, exist_ok=True)

        # Compress images in parallel
        def _compress_and_save(item):
            scan_fn, img_bytes = item
            if detections_only and scan_fn not in pages_with_detections:
                return None
            stem = Path(scan_fn).stem
            jpg_bytes, scale = _compress_image(img_bytes)
            (img_dir / f"{stem}.jpg").write_bytes(jpg_bytes)
            return (scan_fn, scale)

        scale_factors = {}
        exported_pages = 0
        with ThreadPoolExecutor(max_workers=8) as img_pool:
            futures = [img_pool.submit(_compress_and_save, item) for item in images.items()]
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    scale_factors[result[0]] = result[1]
                    exported_pages += 1

        vol_json = _build_volume_json(barcode, vol_meta, rows, scale_factors)

        if detections_only:
            vol_json["scan_filenames"] = [
                fn for fn in vol_json["scan_filenames"] if fn in pages_with_detections
            ]
        else:
            available_stems = {Path(fn).stem for fn in images.keys()}
            vol_json["scan_filenames"] = [
                fn for fn in vol_json["scan_filenames"]
                if Path(fn).stem in available_stems
            ] + [
                fn for fn in images.keys()
                if fn not in vol_json["scan_filenames"] and Path(fn).stem in available_stems
            ]
            seen = set()
            deduped = []
            for fn in vol_json["scan_filenames"]:
                if fn not in seen:
                    seen.add(fn)
                    deduped.append(fn)
            vol_json["scan_filenames"] = deduped

        vol_json_path = output_path / "data" / "volumes" / f"{barcode}.json"
        with open(vol_json_path, "w") as f:
            json.dump(vol_json, f)

        logger.info(f"  [{barcode}] Done — {exported_pages} pages exported")
        return {
            "barcode": barcode,
            "page_count": exported_pages,
            "detection_count": len(rows),
        }

    # Process volumes in parallel (S3 downloads + image compression)
    index_volumes = []
    with ThreadPoolExecutor(max_workers=4) as vol_pool:
        futures = {vol_pool.submit(_process_volume, bc): bc for bc in barcodes}
        for fut in as_completed(futures):
            try:
                index_volumes.append(fut.result())
            except Exception as e:
                bc = futures[fut]
                logger.error(f"  [{bc}] Failed: {e}")

    index_path = output_path / "data" / "index.json"
    with open(index_path, "w") as f:
        json.dump({"volumes": index_volumes}, f, indent=2)

    (output_path / "app.py").write_text(_generate_app_py())
    _copy_static_assets(output_path)
    (output_path / "requirements.txt").write_text(_generate_requirements_txt())
    (output_path / "README.md").write_text(_generate_readme())

    logo_src = Path("data/logo.png")
    if logo_src.exists():
        shutil.copy(logo_src, output_path / "data" / "logo.png")

    for model_file in ["detect.pt", "classify.pt"]:
        model_src = Path("data") / model_file
        if model_src.exists():
            shutil.copy(model_src, output_path / "data" / model_file)
            logger.info(f"  Copied {model_file}")

    logger.info(f"Space exported to {output_path}")
    logger.info(f"  Volumes: {len(barcodes)}")
    logger.info(f"  Total pages: {sum(v['page_count'] for v in index_volumes)}")
    logger.info(f"  Total detections: {sum(v['detection_count'] for v in index_volumes)}")

    if push:
        import os
        logger.info(f"Pushing images to HuggingFace dataset {HF_DATA_REPO}...")
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        images_dir = output_path / "data" / "images"
        if images_dir.exists():
            api.sync_bucket(
                source=str(images_dir),
                dest=f"hf://buckets/{HF_DATA_REPO}/images",
                delete=True,
                token=token,
            )
            logger.info(f"Images pushed to bucket {HF_DATA_REPO}")

        logger.info(f"Pushing Space files to {HF_SPACE_REPO}...")
        if images_dir.exists():
            shutil.rmtree(images_dir)

        logger.info("  Recreating Space repo to free LFS storage...")
        try:
            api.delete_repo(repo_id=HF_SPACE_REPO, repo_type="space")
        except Exception:
            pass
        api.create_repo(
            repo_id=HF_SPACE_REPO,
            repo_type="space",
            space_sdk="gradio",
            private=True,
        )

        api.upload_folder(
            folder_path=str(output_path),
            repo_id=HF_SPACE_REPO,
            repo_type="space",
        )
        logger.info(f"Pushed successfully to https://huggingface.co/spaces/{HF_SPACE_REPO}")
