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
DETECTION_MODEL_REPO = "institutional/institutional-books-visual-elements-detection-yolo26n"
""" Name of the repo containing final weights for the detection model. """

DETECTION_MODEL_FILEPATH = "weights/best.pt"
""" Filepath of the detection model within `DETECTION_MODEL_REPO`. """

DETECTION_MODEL_IMGSZ = 640
""" `imgsz` value to pass to YOLO during inference. """

DETECTION_MODEL_CONF = 0.3
""" `conf` value to pass to YOLO during inference. """

DETECTION_MODEL_IOU = 0.2
""" `iou` value to pass to YOLO during inference. """

DETECTION_MODEL_PROCESSES_PER_GPU = int(os.getenv("DETECTION_MODEL_PROCESSES_PER_GPU", 1))
""" Determines how many detection processes can run on a given GPU. """

#
# Classification model params
#
CLASSIFICATION_MODEL_PROCESSES_PER_GPU = int(os.getenv("CLASSIFICATION_MODEL_PROCESSES_PER_GPU", 1))
""" Determines how many classification processes can run on a given GPU. """

CLASSIFICATION_MODEL_REPO = (
    "institutional/institutional-books-visual-elements-classification-yolo26s-cls"
)
""" Name of the repo containing final weights for the classification model. """

CLASSIFICATION_MODEL_FILEPATH = "weights/best.pt"
""" Filepath of the classification model within `CLASSIFICATION_MODEL_REPO`. """

CLASSIFICATION_MODEL_IMGSZ = 640
""" `imgsz` value to pass to YOLO during inference. """

CLASSIFICATION_MODEL_CONF = 0.0
""" `conf` value to pass to YOLO during inference. """

CLASSIFICATION_MAX_BATCH = 16
""" Maximum batch size to pass into the YOLO classification model """

CLASSIFICATION_CLASS_DICT = {
    "Other": "Other",
    "Image or Illustration": "Image/Illustration",
    "Ex Libris or Decorative": "Ex Libris/Decorative",
    "Music": "Music",
    "Chart or Graph": "Chart/Graph",
    "Artifact": "Artifact",
}
""" Dictionary mapping class to class label """


#
# Dedupe Embeddings
#
DEDUPE_EMBEDDING_MODEL_STORAGE_PATH = "pretrained-models"
""" S3 folder in which the deduplication model is saved. """

DEDUPE_EMBEDDING_MODEL_NAME = "sscd_disc_mixup.torchscript.pt"

DEDUPE_EMBEDDING_MODEL_FILEPATH = Path(
    f"{DEDUPE_EMBEDDING_MODEL_STORAGE_PATH}/{DEDUPE_EMBEDDING_MODEL_NAME}"
)
""" Local filepath to save downloaded embedding model. """

DEDUPE_EMBEDDING_NUM_PROCESSES_PER_GPU = 2
"""Number of processes to run in parallel per GPU"""

DEDUPE_EMBEDDING_BATCH_SIZE = 256
"""Size of minibatches to pass to the embedding model"""

DEDUPE_EMBEDDING_THRESHOLD = 0.14
"""Max cosine distance between embeddings to be considered a duplicate"""

DEDUPE_EMBEDDING_MAX_NEIGHBORS = 100
"""Maximum neighbors to find per embedding (k in HNSW search)"""

DEDUPE_EMBEDDING_MAX_CONNECTIONS = 16
"""HNSW index M parameter (number of connections per layer)"""

DEDUPE_EMBEDDING_HNSW_EF_CONST = 200
"""HNSW ef_construction parameter (index build time, higher=better recall)"""

DEDUPE_EMBEDDING_HNSW_EF_SEARCH = 300
"""HNSW ef_search parameter (query time, higher=better recall)"""

DEDUPE_EMBEDDING_HNSW_INDEX_BATCH = 100000
"""Batch size for index building"""

DEDUPE_EMBEDDING_SEARCH_BATCH = 10000
"""Batch size for similarity search (smaller=less memory)"""

DEDUPE_EMBEDDING_CACHE_DIR = Path(CACHE_DIR_PATH, "temp_image_embeddings")
"""Directory to cache embedding data files"""


#
# Dedupe Hashes
#
HASH_DEDUPE_LENGTH_BYTES = 12  # NOTE: change the max_length of the ImageHash model accordingly
"""Size of phash. The hash will be of size HASH_SIZE*HASH_SIZE bytes"""

HASH_DB_CHUNK_SIZE = 10000
"""Size of chunks to write grouped embeddings to DB"""

HASH_DEDUPE_HAMMING_THRESHOLD = 16
"""Max Hamming distance to be considered duplicate"""

HASH_DEDUPE_LSH_NUM_TABLES = 5
"""Number of LSH hash tables (more = better recall, slower)"""

HASH_DEDUPE_LSH_KEY_SIZE = 8
"""Number of bits per LSH key (smaller = more candidates, slower)"""

HASH_DEDUPE_CACHE_DIR = Path(CACHE_DIR_PATH, "temp_image_hashes")
"""Directory to cache hash data files"""


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

OPENAI_REQUEST_TIMEOUT = 20.0
"""Number of seconds to wait before API request timeout"""

CAPTION_REQUEST_RETRY_ATTEMPTS = 2
"""Number of times to retry caption request before moving on"""

CAPTION_CLASSES_EXCLUDED = ["Ex Libris or Decorative", "Artifact"]
"""Classes to exclude from captioning"""

CAPTION_MAX_BATCH_SIZE = 8
"""Max request batch size to send at once to OpenAI"""

CPUS_LIMIT_CAPTION = 128
""" Default CPU limit for captioning step """


#
# Storage
#
OUTPUT_STORAGE_BUCKET_NAME = str(os.getenv("OUTPUT_BUCKET_NAME"))
"""Bucket name where we store output"""


#
# Analysis
#
ANALYSIS_OUTPUT_DIR = PEEK_OUTPUT_DIR = Path(CACHE_DIR_PATH, "temp_analysis")
"""Output directory for analysis files"""


#
# Misc
#
NODE_NAME = os.getenv("NODE_NAME")
""" Name of the current machine. """

CPUS_LIMIT = int(os.getenv("CPUS_LIMIT", multiprocessing.cpu_count()))
""" Determines how many CPU cores should be used. Loose limit. """

CUDA_GPUS = [device for device in get_torch_devices() if device.startswith("cuda:")]
""" List of currently available CUDA-capable GPUs. """

MAX_S3_CONCURRENCY = int(os.getenv("MAX_S3_CONCURRENCY"))
""" Determines how many operations can be run in parallel against S3-compatible storage. Loose limit. """

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

PEEK_OUTPUT_DIR = Path(CACHE_DIR_PATH, "temp_peek")
"""Output directory for visualization"""

OMP_NUM_THREADS = int(os.getenv("OMP_NUM_THREADS", 1))
"""Make sure PyTorch doesn't use its own thread pool"""
