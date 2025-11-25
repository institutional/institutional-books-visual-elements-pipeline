import os
from pathlib import Path
from datetime import datetime, timezone
import multiprocessing

from slugify import slugify
from dotenv import load_dotenv
from utils.get_torch_devices import get_torch_devices

load_dotenv()

#
# Required env vars
#
REQUIRED_ENV_VARS = [
    "DATA_DIR_PATH",
    "CACHE_MAX_SIZE_IN_GB",
    "NODE_NAME",
    "GRIN_DATA_RUN_NAME",
    "GRIN_DATA_RAW_BUCKET",
    "GRIN_DATA_ENDPOINT",
    "GRIN_DATA_ACCESS_KEY_ID",
    "GRIN_DATA_SECRET_ACCESS_KEY",
    "OUTPUT_BUCKET_NAME",
    "OUTPUT_ENDPOINT",
    "OUTPUT_ACCESS_KEY_ID",
    "OUTPUT_SECRET_ACCESS_KEY",
    "OUTPUT_REGION",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "MAX_S3_CONCURRENCY",
    "MAX_DB_CONCURRENCY",
    "PIPELINE_BATCH_TIMEOUT_SECONDS",
    "HF_TOKEN",
]
""" Lists project-wide required environment variables. """

#
# Data directory
#
DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH", "data/")
""" Data directory: root path. """

CACHE_DIR_PATH = Path(DATA_DIR_PATH, "cache")
""" Data directory: cache """

#
# Input dataset params
#
IB_10_METADATA_DATASET_REPO = "institutional/institutional-books-1.0-metadata"
""" Name of the metadata-only version of Institutional Books 1.0 on Hugging Face. """

#
# YOLO models params
#
DETECTION_MODEL_REPO = "institutional/institutional-books-visual-elements-detection-yolov11n"
""" Name of the repo containing final weights for the detection model. """

DETECTION_MODEL_FILEPATH = "weights/best.pt"
""" Filepath of the detection model within `DETECTION_MODEL_REPO`. """

DETECTION_MODEL_IMGSZ = 640
""" `imgsz` value to pass to YOLO during inference. """

DETECTION_MODEL_CONF = 0.6
""" `conf` value to pass to YOLO during inference. """

DETECTION_MODEL_IOU = 0.3
""" `iou` value to pass to YOLO during inference. """

DETECTION_MODEL_PROCESSES_PER_GPU = int(os.getenv("DETECTION_MODEL_PROCESSES_PER_GPU", 1))
""" Determines how many detection processes can run on a given GPU. """

DETECTION_MODEL_PROCESSES_FORK_DELAY = float(os.getenv("DETECTION_MODEL_PROCESSES_FORK_DELAY", 0.5))
""" 
Sets a delay before creating GPU processes. 
Helpful hack to manage multpiple processes on a single GPU while avoiding collisions.
"""

#
# Classification model (applying class to crop)
#
CLASSIFICATION_MODEL_PROCESSES_PER_GPU = int(os.getenv("CLASSIFICATION_MODEL_PROCESSES_PER_GPU", 1))
""" Determines how many classification processes can run on a given GPU. """

CLASSIFICATION_MODEL_REPO = (
    "institutional/institutional-books-visual-elements-classification-yolo11s-cls"
)
""" Name of the repo containing final weights for the classification model. """

CLASSIFICATION_MODEL_FILEPATH = "weights/best.pt"
""" Filepath of the classification model within `CLASSIFICATION_MODEL_REPO`. """

CLASSIFICATION_MODEL_IMGSZ = 640
""" `imgsz` value to pass to YOLO during inference. """

CLASSIFICATION_MODEL_CONF = 0.25
""" `conf` value to pass to YOLO during inference. """

CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY = float(
    os.getenv("CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY", 0.5)
)
""" 
Sets a delay before creating GPU processes. 
Helpful hack to manage multpiple processes on a single GPU while avoiding collisions.
"""

#
# Dedupe Embeddings
#
DEDUPE_EMBEDDING_MODEL_REPO = "institutional/fork-of-original-repo"
""" Name of the repo containing weights for the dedupe model. """
DEDUPE_EMBEDDING_MODEL_FILEPATH = "sscd_disc_mixup.torchscript.pt"
""" Filepath of the embeddings model within `DEDUPE_EMBEDDING_MODEL_REPO`. """

DEDUPE_EMBEDDING_MODEL_REPO_OWNER = "institutional-data-initiative"

DEDUPE_EMBEDDING_MODEL_REPO_BRANCH = "main"

DEDUPE_EMBEDDING_MODEL_PROCESSES_FORK_DELAY = 0.5
""" 
Sets a delay before creating GPU processes. 
Helpful hack to manage multpiple processes on a single GPU while avoiding collisions.
"""

#
# Captioning
#
MAX_TOKENS_PER_DAY = 14000000000
""" Daily token limit for OpenAI batches."""
CAPTION_MAX_IMG_DIM = 1248
""" Max dimension to send to OpenAI (pixels)"""
CAPTION_MAX_TOKENS = 100
""" Max tokens for model to produce"""
CAPTION_MODEL_NAME = "gpt-4.1-nano"
""" Model to generate captions"""
CAPTION_JSONL_FILES_PATH = "caption_jsonl_files/"
"""WHere to store the jsonl files (temp storage)"""
MAX_REQUESTS_PER_FILE = 5
""" Max requests per OpenAI batch. 
Note: make sure that this produces files less than the size limit (as of Nov 2025, limit is set at 
20 MB per jsonl batch file)"""
# CAPTION_MAX_FILES_PROCESS_PER_DAY = 800000
CAPTION_MAX_FILES_PROCESS_PER_DAY = 100
""" Max requests per day. 
[CAPTION_MAX_FILES_PROCESS_PER_DAY * (avg tokens / input jsonl file)] should be less than the daily token limit"""
CAPTION_BUCKET_NAME = str(os.getenv("OUTPUT_BUCKET_NAME"))

CAPTION_MODEL_TEMPERATURE = 0
CAPTION_TOP_LOGPROBS = 2
CAPTION_MAX_REQUESTS = 1000
"""For budget reasons"""
OPENAI_REQUEST_TIMEOUT = 20.0

MAX_OPENAI_CONCURRENT_REQUESTS = 175

CAPTION_REQUEST_RETRY_ATTEMPTS = 1


#
# Storage
#
BUCKET_NAME = str(os.getenv("OUTPUT_BUCKET_NAME"))

#
# Misc
#
NODE_NAME = os.getenv("NODE_NAME")
""" Name of the current machine. """

CPUS_LIMIT = int(os.getenv("CPUS_LIMIT", multiprocessing.cpu_count()))
""" Determines how many CPU cores should be used. Loose limit. """

CUDA_GPUS = [device for device in get_torch_devices() if device.startswith("cuda:")][
    :4
]  # change later
""" List of currently available CUDA-capable GPUs. """

MAX_S3_CONCURRENCY = int(os.getenv("MAX_S3_CONCURRENCY"))
""" Determines how many operations can be run in parallel against S3-compatible storage. Loose limit. """

MAX_DB_CONCURRENCY = int(os.getenv("MAX_DB_CONCURRENCY"))
""" Determines how many operations can be run in parallel against the database. Loose limit. """

PIPELINE_BATCH_TIMEOUT_SECONDS = int(os.getenv("PIPELINE_BATCH_TIMEOUT_SECONDS"))
""" Maximum amount of time, in seconds, during which any given batch can run. """

DATETIME_SLUG = datetime_slug = slugify(
    datetime.now(timezone.utc).isoformat(sep=" ", timespec="minutes")
)
""" Datetime slug. Hoisted at `const` level for convenience. """

DEFAULT_DB_BATCH_SIZE = 1000
""" 
    Default batch size for database operations.
    Database write/update operations are batched throughout this pipeline.
    This value is used to determine how often the pipeline should attempt to write to the database.
    Applies across the codebase unless specified otherwise.
"""
