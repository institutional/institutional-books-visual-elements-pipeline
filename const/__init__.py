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
# Detectiom model params
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
# Classification model params
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

CLASSIFICATION_MODEL_CONF = 0.90
""" `conf` value to pass to YOLO during inference. """

CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY = float(
    os.getenv("CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY", 0.5)
)
""" 
Sets a delay before creating GPU processes. 
Helpful hack to manage multpiple processes on a single GPU while avoiding collisions.
"""

CLASSIFICATION_MAX_BATCH = 16
""" Maximum batch size to pass into the YOLO classification model """

CLASS_DICT = {
    "0": "Other",
    "1": "Image/Illustration",
    "2": "Ex Libris/Decorative",
    "3": "Advertisement",
    "5": "Music",
    "6": "Handwritten",
    "9": "Diagram/Graph",
    "10": "Artifact",
}
""" Dictionary mapping class to class label """


#
# Dedupe Embeddings/Hashes
#
DEDUPE_EMBEDDING_MODEL_STORAGE_PATH = "pretrained-models"
""" S3 folder in which the deduplication model is saved. """

DEDUPE_EMBEDDING_MODEL_NAME = "sscd_disc_mixup.torchscript.pt"

DEDUPE_EMBEDDING_MODEL_FILEPATH = Path(
    f"{DEDUPE_EMBEDDING_MODEL_STORAGE_PATH}/{DEDUPE_EMBEDDING_MODEL_NAME}"
)
""" Local filepath to save downloaded embedding model. """

DEDUPE_EMBEDDING_MODEL_PROCESSES_FORK_DELAY = 0.5
""" 
Sets a delay before creating GPU processes. 
Helpful hack to manage multpiple processes on a single GPU while avoiding collisions.
"""

HASH_SIZE = 12  # NOTE: change the max_length of the ImageHash model accordingly
"""Size of phash. The hash will be of size HASH_SIZE*HASH_SIZE bytes"""

DEDUPE_EMBEDDING_BATCH_SIZE = 1024
"""Size of minibatches to pass to the embedding model"""

HASH_DB_CHUNK_SIZE = 10000
"""Size of chunks to write grouped embeddings to DB"""

HASH_DEDUPE_HAMMING_THRESHOLD = 16
"""Max Hamming distance to be considered duplicate"""

HASH_DEDUPE_TOTAL_SHARDS = 10
"""Total number of shards to split dataset into when grouping"""

HASH_DEDUPE_MAX_HASHES_PER_SHARD = 1000
"""Maximum hashes per shard (auto-shard if exceeded)"""


#
# Captioning
#
CAPTION_MAX_IMG_DIM = 1248
""" Max dimension to send to OpenAI (pixels)"""

CAPTION_MAX_TOKENS = 100
""" Max tokens for model to produce"""

CAPTION_MODEL_NAME = "gpt-4.1-nano"
""" Model to generate captions"""

CAPTION_MODEL_TEMPERATURE = 0
"""Temperature setting for OpenAI model"""

CAPTION_TOP_LOGPROBS = 2
"""Number of logprobs to retrieve for each token from OpenAI model"""

CAPTION_MAX_REQUESTS = 1000
"""Maximum caption requests to process (For budget reasons)"""

OPENAI_REQUEST_TIMEOUT = 20.0
"""Number of seconds to wait before API request timeout"""

CAPTION_REQUEST_RETRY_ATTEMPTS = 2
"""Number of times to retry caption request before moving on"""

CAPTION_CLASSES_EXCLUDED = ["2", "6", "10"]
"""Classes to exclude from captioning"""

CAPTION_MAX_BATCH_SIZE = 8
"""Max request batch size to send at once to OpenAI"""


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

MAX_S3_REQUESTS_PER_SECOND = 25
"""Maximum upload requests to S3 per second (for step05_store)"""
