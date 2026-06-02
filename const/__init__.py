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
    "FILTER_BUCKET_NAME",
    "FILTER_ENDPOINT",
    "FILTER_ACCESS_KEY_ID",
    "FILTER_SECRET_ACCESS_KEY",
    "FILTER_REGION",
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

DEDUPE_EMBEDDING_MAX_NEIGHBORS = 30
"""Maximum neighbors to find per embedding (k in HNSW search)"""

DEDUPE_EMBEDDING_MAX_CONNECTIONS = 12
"""HNSW index M parameter (number of connections per layer)"""

DEDUPE_EMBEDDING_HNSW_EF_CONST = 100
"""HNSW ef_construction parameter (index build time, higher=better recall)"""

DEDUPE_EMBEDDING_HNSW_EF_SEARCH = 30
"""HNSW ef_search parameter (query time, higher=better recall)"""

DEDUPE_EMBEDDING_HNSW_INDEX_BATCH = 100000
"""Batch size for index building"""

DEDUPE_EMBEDDING_SEARCH_BATCH = 50000
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

HASH_DEDUPE_CPUS_LIMIT = 64
"""Limit the number of CPUs for hash deduping"""

HASH_DEDUPE_BANDS = 6
"""Bands for hash dedupe"""

HASH_DEDUPE_BITS_PER_BAND = (
    HASH_DEDUPE_LENGTH_BYTES * HASH_DEDUPE_LENGTH_BYTES
) // HASH_DEDUPE_BANDS
"""calculate bits per band based on hash size"""

HASH_DEDUPE_MASK_24 = (1 << 24) - 1
"""Mask for hash dedupe"""

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

CPUS_LIMIT_CAPTION = 20
""" Default CPU limit for captioning step """


#
# Storage
#
OUTPUT_STORAGE_BUCKET_NAME = str(os.getenv("OUTPUT_BUCKET_NAME"))
"""Bucket name where we store output"""

FILTER_STORAGE_BUCKET_NAME = str(os.getenv("FILTER_BUCKET_NAME"))
"""Bucket name where we store filtered dataset exports"""


#
# Analysis
#
ANALYSIS_OUTPUT_DIR = PEEK_OUTPUT_DIR = Path(CACHE_DIR_PATH, "temp_analysis")
"""Output directory for analysis files"""

#
# Exporting
#
DETECTION_CONFIDENCE_THRESHOLD = 0.75
""" Minimum detection confidence to include a crop in the export. Detections below this are excluded. """

CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.70
""" Classifications below this confidence are reclassified as "Other" in the export. """

SERVER_SIDE_CURSOR_SIZE = 100_000
""" Number of rows to fetch per batch when streaming results via PostgreSQL server-side cursors. """

MODEL_CLASS_INDEX_ORDER = [
    "Artifact",
    "Chart/Graph",
    "Ex Libris/Decorative",
    "Image/Illustration",
    "Music",
]
""" Ordered list of classification labels matching the YOLO model's class index positions. """

S3_ROW_GROUP_SIZE = 500
""" Number of records per Parquet row group. Kept small to limit memory from crop bytes. """

S3_MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100 MB
""" File size above which S3 uploads switch to multipart. """

S3_MULTIPART_CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB
""" Size of each part in a multipart S3 upload. """

S3_MULTIPART_PARALLEL_PARTS = 6
""" Number of parts to upload to S3 concurrently during a multipart S3 upload. """

S3_BATCH_SIZE_DB = 500_000
""" Batch size for PostgreSQL IN-clause queries to avoid exceeding memory limits. """

S3_SAMPLE_LIMIT = 10_000
""" Maximum number of records to export to S3 when running in sample mode. """

S3_MAX_PENDING_UPLOADS = 8
""" Maximum number of concurrent shard uploads to S3 before backpressure pauses record processing. """

S3_MAX_INFLIGHT = 4
""" Maximum number of S3 item download tasks in flight at once during export. """

S3_SHARD_SIZE = 5000
""" Default number of rows per Parquet shard file uploaded to S3. """

HF_IMAGES_REPO = "institutional/institutional-books-hl-visual-elements-images"
""" Hugging Face repo for uploading crop images as a bucket. """

HF_DATASET_REPO = "institutional/institutional-books-hl-visual-elements"
""" Hugging Face repo for the parquet dataset (metadata + embeddings). """

HF_SAMPLE_LIMIT = 500
""" Maximum number of items to export when running in sample mode. """

HF_NETWORK_TIMEOUT = 300
""" Seconds to wait before timing out a Hugging Face network operation. """

HF_NETWORK_MAX_RETRIES = 5
""" Maximum number of retry attempts for failed Hugging Face network calls. """

HF_NETWORK_BASE_DELAY = 5
""" Base delay in seconds for exponential backoff between retries. """

HF_ITEM_IDS_CACHE_PATH = Path(ANALYSIS_OUTPUT_DIR) / "hf_export_item_ids.json"
""" Local cache file storing item IDs to resume interrupted HF exports. """

HF_SHARD_SIZE = 5000
""" Number of rows per parquet shard uploaded to the HF dataset repo. """

HF_IMAGE_BATCH_SIZE = 1000
""" Number of images to upload per commit to the HF images bucket. """

HF_ITEMS_PER_FETCH = 200
""" Number of pipeline batch items to fetch from the database per query. """

HF_IO_WORKERS = 4
""" Number of parallel I/O workers for downloading crops from S3. """

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
