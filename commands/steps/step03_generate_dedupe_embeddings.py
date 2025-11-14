import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
import gc
import time

import click
from loguru import logger
import cv2
import numpy as np

from more_itertools import chunked
import imagehash

from utils import get_db, process_db_write_batch
from models import PipelineBatchItem, Detection, Embedding, ImageHash

from const import (
    DEDUPE_EMBEDDING_MODEL_REPO,
    DEDUPE_EMBEDDING_MODEL_FILEPATH,
    DEDUPE_EMBEDDING_MODEL_REPO_OWNER,
    DEDUPE_EMBEDDING_MODEL_REPO_BRANCH,
    CUDA_GPUS,
    CPUS_LIMIT,
)
import requests
import os


@click.command("step03-embed")
@click.option(
    "--id-pipeline-batch",
    type=int,
)
@click.option(
    "--cpus-limit",
    type=int,
    default=CPUS_LIMIT,
    help="Allows for limiting the number of CPU cores this command can use.",
)
@click.option(
    "--cuda-gpus",
    type=click.Choice(CUDA_GPUS),
    multiple=True,
    required=True,
    default=CUDA_GPUS if CUDA_GPUS else ["cuda:0"],
    help="Determines on which specific CUDA device(s) this command should use.",
)
def step03_generate_dedupe_embeddings(
    id_pipeline_batch: int, cpus_limit: int, cuda_gpus: list[str]
):
    """
    Computes embeddings (and hashes) for all crops in all volumes with detections in this pipeline batch,
    and saves them to the database, per GPU.
    """
    model_filepath: Path | None = None
    cuda_gpus_total = len(cuda_gpus)
    processes_total = cuda_gpus_total

    item_id_batches: list[list[int]] = [[] for _ in range(processes_total)]

    per_task_cpus_limit = int(round(cpus_limit / cuda_gpus_total))
    if processes_total > 1:
        per_task_cpus_limit = max(2, per_task_cpus_limit // 2)

    local_model_path = Path("pretrained", os.path.basename(DEDUPE_EMBEDDING_MODEL_FILEPATH))
    if not local_model_path.exists():
        logger.info(f"Downloading TorchScript model from GitHub ...")
        download_github_file(
            repo_owner=DEDUPE_EMBEDDING_MODEL_REPO_OWNER,
            repo_name=DEDUPE_EMBEDDING_MODEL_REPO,
            file_path=DEDUPE_EMBEDDING_MODEL_FILEPATH,
            branch=DEDUPE_EMBEDDING_MODEL_REPO_BRANCH,
            destination_path=local_model_path,
        )
    else:
        logger.info(f"Model file found at {local_model_path}, skipping download.")
    model_filepath = local_model_path

    # Only process volumes with detections
    eligible_items_query = (
        PipelineBatchItem.select(PipelineBatchItem)
        .where(
            (PipelineBatchItem.pipeline_batch == id_pipeline_batch)
            & PipelineBatchItem.id_pipeline_batch_item.in_(
                Detection.select(
                    Detection.pipeline_batch_item
                )  # Only volumes with at least 1 detection
            )
        )
        .order_by(PipelineBatchItem.id_pipeline_batch_item)
        .distinct()
    )
    eligible_items = list(eligible_items_query)
    for i, item in enumerate(eligible_items):
        process_i = i % processes_total
        item_id_batches[process_i].append(item.id_pipeline_batch_item)

    if not any(item_id_batches):
        logger.warning("No eligible items with detections found for this batch. Exiting.")
        click.get_current_context().exit(0)

    with ProcessPoolExecutor(max_workers=processes_total, initializer=get_db) as executor:
        futures = {}
        for i, item_ids in enumerate(item_id_batches):
            cuda_gpus_i = i % cuda_gpus_total
            future = executor.submit(
                embed_batch_of_items,
                item_ids=item_ids,
                model_filepath=model_filepath,
                cuda_device=cuda_gpus[cuda_gpus_i],
                cpus_limit=per_task_cpus_limit,
            )
            futures[future] = cuda_gpus[cuda_gpus_i]
            time.sleep(0.5)  # mimic detection/classification fork delay
        for future in as_completed(futures):
            cuda_gpu: str = futures[future]
            try:
                future.result()
            except Exception as err:
                logger.debug(traceback.print_exc())
                logger.error(
                    f"A blocking error occured while embedding batch on {cuda_gpu}. Exiting."
                )
                executor.shutdown(wait=False, cancel_futures=True)
                click.get_current_context().exit(1)
            except KeyboardInterrupt as err:
                logger.warning("Received interrupt signal")
                raise err


def embed_batch_of_items(
    item_ids: list[int],
    model_filepath: Path,
    cuda_device: str,
    cpus_limit: int,
):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device.replace("cuda:", "")
    import torch
    from PIL import Image

    device = "cuda:0"

    # Load TorchScript model only ONCE per process
    model = torch.jit.load(str(model_filepath), map_location=device)
    model.eval()

    def preprocess_for_model(crop: np.ndarray):
        img = Image.fromarray(crop.astype(np.uint8))
        img = img.convert("RGB").resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img)  # [C, H, W]

    from datetime import datetime, timezone

    for id_pipeline_batch_item in item_ids:
        item = PipelineBatchItem.get(id_pipeline_batch_item=id_pipeline_batch_item)
        volume_barcode = item.ib_volume.barcode

        item_detections = (
            Detection.select()
            .where(Detection.pipeline_batch_item == id_pipeline_batch_item)
            .order_by(Detection.id_detection)
        )

        if item_detections.count() == 0:
            logger.info(f"{volume_barcode}: No detections - skipping embedding for this item.")
            continue

        image_bytes_by_filename = dict(list(item.data.images.items()))
        image_bytes_by_filename = {str(k): v for k, v in image_bytes_by_filename.items()}

        # 1. Decode all scans needed for this item
        with ThreadPoolExecutor(max_workers=cpus_limit) as decode_executor:
            futures = {}
            used_filenames = set(str(det.scan_filename) for det in item_detections)
            loaded_images: dict[str, np.ndarray] = {}
            for fn in used_filenames:
                if fn not in image_bytes_by_filename:
                    logger.warning(
                        f"Missing image bytes for scan {volume_barcode}.{fn} - skipping this scan in embedding"
                    )
                    continue
                futures[decode_executor.submit(decode_image_bytes, image_bytes_by_filename[fn])] = (
                    fn
                )
            done, _ = wait(futures)
            for future in done:
                fn = futures[future]
                try:
                    loaded_images[fn] = future.result()
                except Exception:
                    logger.warning(f"Could not decode scan {volume_barcode}.{fn}")

        embedding_entries = []
        imagehash_entries = []
        n_embeds, failed_embeds = 0, 0

        # Compute embedding and hash for each crop (per detection)
        crops_and_meta = []
        for det in item_detections:
            scan_img = loaded_images.get(str(det.scan_filename))
            if scan_img is None:
                failed_embeds += 1
                continue
            try:
                crop = det.crop(scan_img)
                crops_and_meta.append((det, crop, str(det.scan_filename)))
                n_embeds += 1
            except Exception:
                logger.warning(
                    f"Could not crop detection in {volume_barcode}.{det.scan_filename}; skipping"
                )
                failed_embeds += 1

        # Prepare model inputs in minibatches
        batch_size = 1024
        crop_batches = list(chunked(crops_and_meta, batch_size))

        for batch in crop_batches:
            detections_batch, crops_batch, filenames_batch = zip(*batch)
            # Prep
            prepped = [preprocess_for_model(crop) for crop in crops_batch]
            batch_tensor = torch.stack(prepped, dim=0).to(device)
            # Embedding inference
            with torch.no_grad():
                embeds = model(batch_tensor)  # [B, 512]
            embeds = embeds.cpu().numpy()
            # Normalize
            embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)

            for idx, det in enumerate(detections_batch):
                embedding_entries.append(
                    Embedding(
                        detection_id=det.id_detection,
                        pipeline_batch_item=id_pipeline_batch_item,
                        scan_filename=filenames_batch[idx],
                        embedding=embeds[idx].tolist(),
                        created=datetime.now(timezone.utc),
                    )
                )
                # Hash (pHash)
                crop_img_pil = Image.fromarray(crops_batch[idx].astype(np.uint8))
                h = imagehash.phash(crop_img_pil, hash_size=8)
                imagehash_val = str(h)  # hex string (e.g. 'feaf3452aaa21344')
                imagehash_entries.append(
                    ImageHash(
                        detection_id=det.id_detection,
                        pipeline_batch_item=id_pipeline_batch_item,
                        scan_filename=filenames_batch[idx],
                        image_hash=imagehash_val,
                        created=datetime.now(timezone.utc),
                    )
                )

        # Store in DB (replace previous for this batch item)
        Embedding.delete().where(Embedding.pipeline_batch_item == id_pipeline_batch_item).execute()
        ImageHash.delete().where(ImageHash.pipeline_batch_item == id_pipeline_batch_item).execute()
        process_db_write_batch(
            model=Embedding,
            entries_to_create=embedding_entries,
        )
        process_db_write_batch(
            model=ImageHash,
            entries_to_create=imagehash_entries,
        )

        logger.info(
            f"{volume_barcode} | n_crops: {n_embeds} - failed crops: {failed_embeds} - embeddings: {len(embedding_entries)} - hashes: {len(imagehash_entries)}"
        )
        # GC/CUDA clear
        torch.cuda.empty_cache()
        gc.collect()

    return True


def decode_image_bytes(image_bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR_RGB)


def download_github_file(repo_owner, repo_name, file_path, branch="main", destination_path=None):
    if destination_path is None:
        destination_path = os.path.basename(file_path)
    raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}"
    response = requests.get(raw_url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            f.write(response.content)
        logger.info(f"File '{file_path}' downloaded to '{destination_path}' successfully.")
    else:
        raise Exception(
            f"Failed to download file. Status code: {response.status_code}\nURL was: {raw_url}"
        )
