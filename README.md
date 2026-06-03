# 📚👁 Institutional Books Visual Elements Pipeline
The Institutional Data Initiative's pipeline for extracting, analyzing, and publishing visual elements from the Institutional Books 1.0 collection.

- 🤗 [Institutional Books on HuggingFace](https://huggingface.co/collections/instdin/institutional-books-68366258bfb38364238477cf)
- 📄 [Technical report](TODO)
- 🌐 [Website](TODO)

---

## Summary 
- [Getting started](#getting-started)
- [Concept](#concept)
- [Sequencing of a pipeline run](#sequencing-of-a-pipeline-run)
- [Available utilities](#available-utilities)
- [CLI: `system`](#cli-system)
- [CLI: `orchestration`](#cli-orchestration)
- [CLI: `steps`](#cli-steps)
- [CLI: `export`](#cli-export)
- [CLI: `post-processing`](#cli-post-processing)
- [About IDI](#about-idi)
- [Cite](#cite)

---

## Getting started 

### Machine-level dependencies
- [uv](https://docs.astral.sh/uv/)
- Libraries to encode/decode JPEG, TIFF and JP2 files.

**Recommended:** A tool for inspecting and exporting the underlying database, such as [pgAdmin4](https://www.pgadmin.org/).

### External dependencies
- A [PostgreSQL database](https://www.postgresql.org/), which can be local or remote.
  - [pgvector extension](https://github.com/pgvector/pgvector)
- An S3-compatible storage bucket for storing intermediary outputs

### Other requirements
- At least 1 CUDA-compatible GPU with 24Gb VRAM or more
- At least 128Gb of RAM
- At least 8 CPU cores

### GRIN Transfer

This pipeline was built and optimized to process a Google Books collection retrieved using [GRIN Transfer](https://github.com/institutional/grin-transfer) and stored on cloud storage.

See [GRIN Transfer](https://github.com/institutional/grin-transfer) documentation for details.

### Step by step setup
```bash
# Clone project
git clone https://github.com/instdin/institutional-books-1-ve-pipeline.git

# Install dependencies
# NOTE: Will attempt to install system-level dependencies on Debian-based systems.
bash install.sh

# Edit environment variables
# We recommend increasing `CACHE_MAX_SIZE_IN_GB` significantly if possible (e.g: 1000 Gb)
nano .env # (or any text editor)

# Run commands
uv run pipeline.py command options
uv run pipeline.py --verbose command options # Run command and include debug logs
```

### Command groups
- **system**: Commands related to system setup.
- **orchestration**: Commands related to pipeline runs.
- **steps**: Individual steps. Are meant to be run by the orchestration scripts.
- **export**: Commands related to the export of both intermediary data and complete datasets.
- **post-processing**: Commands for post-processing operations on pipeline data (backfill, token counting, embedding visualization).

**NOTE:** 
- All commands come with a `--help` flag to list options.
- We recommend storing logs individually, as such: 
```bash
uv run pipeline.py --verbose command options &> run-{date}-{info}.log
```

[👆 Back to the summary](#summary)

---

## Concept

**This pipeline was assembled with the following constraints in mind:**
- The need to run inference using a series of small models against hundreds of millions of high-resolution scans
- Balancing these two bottlenecks: 
  - Loading and pre-processing images at scale
  - Running inference against these images at scale
- The need to scale a pipeline run horizontally if required with as little friction as possible

**As such, it is centered around the following principleds:**
- **Centralized database and object storage**: Allows multiple machines to work on different portions of a pipeline run.
- **Central orchestration:** A central command orchestrates the execution of a pipeline run, which was prepared ahead of time. This orchestrator:
  - Determines which batches needs to be processed using a locking mechanism.
  - Caches the underlying volume-level data of a given batch on disk (scans and OCR-extracted text) before running processing steps against it. This ensures that batch-level steps can quickly and easily access volume-level data. 
  - Makes the distinction between batch-level and run-level steps, the former needing full access to volume data, the latter needing access to collection-wide data.

[👆 Back to the summary](#summary)

---

## Sequencing of a pipeline run


### 1. Build the database

The following command needs to be at least once when setting up the database. 
It pulls data from the [metadata-only version of the Institutional Books 1.0 dataset](https://huggingface.co/datasets/institutional/institutional-books-1.0-metadata) and index all volumes from which visual elements needs to be extracted.

A copy of the metadata of each volume is kept at database level so it can be used as part of this pipeline.

```bash
uv run pipeline.py system build
```

### 2. Prepare a pipeline run

The following command will prepare a pipeline run: a series of batches of volumes that need to be processed. 
At the end of this process, the pipeline will return an `id-pipeline-run`, which can then be used to start the process.

```bash
uv run pipeline.py orchestration prepare

# Only include elements that are not part of any other pipeline run
# Can be helpful to process volumes recently added to the collection
uv run pipeline.py orchestration --append-mode

# Specify number of volumes per batches
uv run pipeline.py orchestration prepare --items-per-batch=500

# Prepare a run on a subset
uv run pipeline.py orchestration prepare --offset=50000 --limit=10000
```

### 3. Start a pipeline run

The following command goes through an entire pipeline run both:
- Batch-level steps, which require access to Google Books scan.
- Run-level steps, which require access to collection-level data but not access to scans.

```bash
uv run pipeline.py orchestration execute --id-pipeline-run=1

# Run a specific batch
uv run pipeline.py orchestration execute --id-pipeline-run=1 --force-id-pipeline-batch=45

# Ignore batch locks (batch running on another machine or stalled)
uv run pipeline.py orchestration execute --id-pipeline-run=1 --ignore-locks

# Only run batch-level steps
uv run pipeline.py orchestration execute --id-pipeline-run=1 --batch-processing-only"
```

### 4. Create the `filtered_dataset` view

Before running export commands, you must create a `filtered_dataset` PostgreSQL view. This view joins pipeline data into a single queryable surface that all export commands read from.

The view applies the following filtering logic:
- Filters detections by detection confidence threshold (0.75)
- Applies classification confidence thresholding (low-confidence predictions become "Other")
- Only includes records present in both deduplication groups (hash-based and embedding-based)

**Conceptual structure:**

```sql
CREATE VIEW filtered_dataset AS
SELECT
    d.id_detection, pbi.id_pipeline_batch_item AS pipeline_batch_item_id,
    v.barcode, d.scan_filename,
    d.bbox_xyxy, d.bbox_xywh, d.bbox_conf,
    c.pred_class, c.classification_conf, c.classification_probs,
    cap.text AS caption_text, cap.lang AS caption_lang,
    cap.lang_detected AS caption_lang_detected,
    cap.linear_prob AS caption_linear_prob,
    cap.thesaurus_matches AS caption_thesaurus_matches,
    ih.image_hash, ie.embedding
FROM detection d
JOIN pipeline_batch_item pbi ON ...
JOIN ib_volume v ON ...
JOIN classification c ON ...
LEFT JOIN caption cap ON ...
LEFT JOIN image_hash ih ON ...
LEFT JOIN image_embedding ie ON ...
JOIN deduped_hash dh ON ...
JOIN deduped_embedding de ON ...
WHERE d.bbox_conf >= 0.75;
```

Adapt the exact join conditions and column names to your schema. The view must be created before any export commands can run.

### 5. Export

Once the pipeline run is complete and the `filtered_dataset` view has been created, use export commands to publish results.

**Backfill computed columns** (run before exports that need `lang_detected`, `linear_prob`, or `thesaurus_matches`):

```bash
uv run pipeline.py post-processing backfill
uv run pipeline.py post-processing backfill --cpus-limit 8
uv run pipeline.py post-processing backfill --skip-thesaurus  # Skip ChronAm thesaurus (no HF_TOKEN needed)
```

> **Note:** The ChronAm thesaurus matching step is optional. Use `--skip-thesaurus` to skip it if you don't need `thesaurus_matches` or don't have a `HF_TOKEN` configured for the HuggingFce Repo. The `lang_detected` and `linear_prob` columns will still be computed.

**Count tokens** in captions:

```bash
uv run pipeline.py post-processing count-tokens
```

**Peek** at samples to confirm the pipeline is working as expected:

```bash
uv run pipeline.py export peek --scope deduplication --id-pipeline-batch 1 --n 10
uv run pipeline.py export peek --scope detection --id-pipeline-batch 1 --n all
```

**Export to S3** (parquet shards with embedded PNG crops):

```bash
# Single process
uv run pipeline.py export to-s3

# Parallel using GNU parallel (recommended for large datasets)
seq 0 31 | parallel -j8 'uv run pipeline.py export to-s3 --chunk-index {} --total-chunks 32'
```

**Export to HuggingFace** (images + parquet datasets):

```bash
# Single process
uv run pipeline.py export to-hf

# Parallel using GNU parallel (recommended for large datasets)
seq 0 31 | parallel -j8 'uv run pipeline.py export to-hf --chunk-index {} --total-chunks 32'
```

**Statistics and visualization:**

```bash
# Aggregate stats and charts
uv run pipeline.py export stats

# Create a HuggingFace Space viewer
uv run pipeline.py export viewer-space --push

# Interactive embedding visualization
uv run pipeline.py post-processing embedding-atlas --sample 10000
```


[👆 Back to the summary](#summary)

---

## Available utilities

Here is an example of some of the utility features this pipeline provides.

```python
from dotenv import load_dotenv
load_dotenv()

import utils
from models import PipelineRun, PipelineBatch, PipelineBatchItem
from models.pipeline_batch_item import PipelineBatchItemData

# We use Peewee as an ORM for this project.
# See Peewee's documentation for more info on how to work with models: https://docs.peewee-orm.com/en/latest/
# See /models/ for more info on which database models are available.

# (Nested as way of clarifying that example, not recommended use)
for run in PipelineRun.select().iterator():
    # Utility: grab all batches for a given run
    for batch in run.batches:
        # Utility: grab all items for a given batch
        for item in batch.item: 
            # Utility: grab data (images and ocr-extracted text) for a given item. 
            # Will retrieve from cache if available, will pull from remote storage otherwise
            data: PipelineBatchItemData = item.data 

            images: list[dict[str, bytes]] = data.images

            text: list[dict[str, str]] = data.text

# Utility: caching volume-level data for an entire batch
# NOTE: Parallelization increases pressure on remote storage
PipelineBatch.get(id_pipeline_batch=1).cache_data(max_workers=16)

# Quick access to the Peewee db connector itself
db = utils.get_db()

# Quick access to the list of available Torch devices.
# Results from the initial pull are memoized.
torch_devices: list[str] = utils.get_torch_devices()

# Quick access to S3 clients connected to role-specific remote storages
s3_grin_data = utils.get_s3_client("GRIN_DATA")
s3_output = utils.get_s3_client("OUTPUT")
```

[👆 Back to the summary](#summary)

---

## CLI: system

> ⚠️ `system build` must be run at least once per database.

<details>
<summary><h3>system build</h3></summary>

Prepares the pipeline by indexing the list of volumes that need to be processed from the metadata-only version of the IB 1.0 dataset.
Stores the barcode and metadata of each volume in the database.

```bash
uv run pipeline.py system build
uv run pipeline.py system build --max-workers=32 # Increases pressure on the database
```
</details>

<details>
<summary><h3>system clear-cache</h3></summary>

Clears disk cache.

```bash
uv run pipeline.py system clear-cache
```

</details>

<details>
<summary><h3>system status</h3></summary>

Reports on the pipeline's status (database and cache size, etc ...)

```bash
uv run pipeline.py system status
```

</details>

[👆 Back to the summary](#summary)

---

## CLI: orchestration

<details>
<summary><h3>orchestration prepare</h3></summary>

Creates a pipeline run and its batches.
This command returns an identifier that can be then passed to `orchestraction execute` to launch a run.

```bash
uv run pipeline.py orchestration prepare

# Only include elements that are not part of any other pipeline run
# Can be helpful to process volumes recently added to the collection
uv run pipeline.py orchestration --append-mode

# Specify number of volumes per batches
uv run pipeline.py orchestration prepare --items-per-batch=500

# Prepare a run on a subset
uv run pipeline.py orchestration prepare --offset=50000 --limit=10000
```

</details>

<details>
<summary><h3>orchestration execute</h3></summary>

Executes a pipeline run.
Runs all steps against all batches unless instructed otherwise.

```bash
uv run pipeline.py orchestration execute --id-pipeline-run=1

# Run a specific batch
uv run pipeline.py orchestration execute --id-pipeline-run=1 --force-id-pipeline-batch=45

# Ignore batch locks (batch running on another machine or stalled)
uv run pipeline.py orchestration execute --id-pipeline-run=1 --ignore-locks

# Only run batch-level steps
uv run pipeline.py orchestration execute --id-pipeline-run=1 --batch-processing-only"
```

</details>

<details>
<summary><h3>orchestration status</h3></summary>

Reports on the status of pipeline runs and associated batches.

```bash
uv run pipeline.py orchestration status
```

</details>

[👆 Back to the summary](#summary)

---

## CLI: steps

<details>
<summary><h3>steps step01-detect</h3></summary>

Runs the visual elements detection model against a batch of volumes.

NOTE:
- This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
- This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.
- Runs X processes per GPU.
    - Adjust `DETECTION_MODEL_PROCESSES_PER_GPU` env var based on available resources.
    - Adjust `DETECTION_MODEL_PROCESSES_FORK_DELAY` env var to adjust pre-fork delay.
    This may help prevent processes from blocking each each other (HACK).

```bash
uv run pipeline.py steps step01-detect --id-pipeline-batch=1
```

</details>

<details>
<summary><h3>steps step02-classify</h3></summary>

Runs the visual elements classification model against a batch of crops.

NOTE:
- This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
- This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.
- Runs X processes per GPU.
    - Adjust `CLASSIFICATION_MODEL_PROCESSES_PER_GPU` env var based on available resources.
    - Adjust `CLASSIFICATION_MODEL_PROCESSES_FORK_DELAY` env var to adjust pre-fork delay.
    This may help prevent processes from blocking each each other (HACK).

```bash
uv run pipeline.py steps step02-classify --id-pipeline-batch=1
```

</details>

<details>
<summary><h3>steps step03-embed</h3></summary>

Computes embeddings (and hashes) for all crops in all volumes with detections in this pipeline batch, and saves them to the database, per GPU.

NOTE:
- This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
- This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.

```bash
uv run pipeline.py steps step03-embed --id-pipeline-batch=1
```

</details>

<details>
<summary><h3>steps step04-process-caption-requests</h3></summary>

Runs caption-generation on the cropped regions of each volume that contains detections.

NOTE:
- This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
- This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.
- Adjust `CAPTION_MAX_REQUESTS` env var based on your OpenAI API tier and usage.


```bash
uv run pipeline.py steps step04-process-caption-requests --id-pipeline-batch=1
```

</details>

<details>
<summary><h3>steps step05-store</h3></summary>

Stores cropped detection regions to S3/R2 storage in full resolution PNG format.

NOTE:
- This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
- This is a batch-level step, which expects to process a batch for which images and text are already cached on disk.

```bash
uv run pipeline.py steps step05-store --id-pipeline-batch=1
```

</details>

<details>
<summary><h3>steps step06-dedupe-fast</h3></summary>

Deduplicate image hashes using external-sort LSH with mmap bucket processing.

Uses a fixed LSH band structure (6 bands x 24 bits for 144-bit perceptual hashes) to identify candidate pairs, then verifies matches using Hamming distance. Processing pipeline:

1. Loads all hashes from DB into binary files (`hashes.bin`, `hash_ids.bin`, `metadata.jsonl`)
2. Generates band entries (TSV) for all hashes across 6 LSH bands
3. External-sorts the band file using GNU `sort`
4. Streams sorted buckets, filters oversized buckets (>20k entries)
5. Processes candidate pairs in parallel using `ProcessPoolExecutor` with `fork` start method
6. Workers read hash data from mmap shared memory (zero-copy via fork inheritance)
7. Builds Union-Find clusters and writes dedupe assignments to DB in parallel

NOTE:
- This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
- This is a run-level step, which expects a `pipeline_run` rather than a `pipeline_batch`.
- Requires GNU `sort` for the external sort step.

```bash
uv run pipeline.py steps step06-dedupe-fast --id-pipeline-run=1

# Custom hamming threshold
uv run pipeline.py steps step06-dedupe-fast --id-pipeline-run=1 --hamming-threshold=12

# Limit workers
uv run pipeline.py steps step06-dedupe-fast --id-pipeline-run=1 --workers=32
```

</details>


<details>
<summary><h3>steps step07-dedupe-embeddings</h3></summary>

Deduplicate embeddings at scale using HNSW index with disk caching.

NOTE: Embeddings are NOT stored in deduped_embedding table (use JOIN with embedding table).

NOTE:
- This command is intended to be run by the orchestrator. See `orchestration/execute.py` for details.
- This is a run-level step, which expects a pipeline_run rather than a pipeline_batch.

```bash
uv run pipeline.py steps step07-dedupe-embeddings --id-pipeline-run=1
```

</details>

[👆 Back to the summary](#summary)

---

## CLI: export

> Export commands require the `filtered_dataset` view to exist in PostgreSQL. See [Create the filtered_dataset view](#4-create-the-filtered_dataset-view).

<details>
<summary><h3>export to-s3</h3></summary>

Export filtered dataset to S3 as parquet shards with embedded PNG crops.

Reads from the `filtered_dataset` view, downloads crops from the OUTPUT S3 bucket (tar.gz archives), and writes parquet files with the crop bytes embedded directly in each row. Supports resume (detects existing shards and skips processed records) and multipart upload for large files.

**Designed for GNU parallel** using `--chunk-index` and `--total-chunks` to split the item list into non-overlapping ranges:

```bash
# Single process
uv run pipeline.py export to-s3

# Parallel (recommended for large datasets)
seq 0 31 | parallel -j8 'uv run pipeline.py export to-s3 --chunk-index {} --total-chunks 32'

# With options
uv run pipeline.py export to-s3 --shard-size 5000 --io-workers 16 --prefix my-export
uv run pipeline.py export to-s3 --sample 100  # Test with 100 items
```

Each chunk writes shards with a unique prefix (e.g., `{prefix}-c05-0001.parquet`).

</details>

<details>
<summary><h3>export to-hf</h3></summary>

Export filtered dataset to HuggingFace: crop images to a HF bucket, metadata to a dataset repo as parquet shards split by classification label.

Reads from the `filtered_dataset` view, downloads crops from the OUTPUT S3 bucket, re-encodes as WebP (quality 95), uploads images via `batch_bucket_files`, and writes parquet shards.

**Designed for GNU parallel** using `--chunk-index` and `--total-chunks`:

```bash
# Single process
uv run pipeline.py export to-hf

# Parallel (recommended for large datasets)
seq 0 31 | parallel -j8 'uv run pipeline.py export to-hf --chunk-index {} --total-chunks 32'

# With options
uv run pipeline.py export to-hf --shard-size 5000 --image-batch-size 200
uv run pipeline.py export to-hf --skip-existing         # Skip items already in HF bucket
uv run pipeline.py export to-hf --skip-parquet-upload   # Images only, combine shards later
uv run pipeline.py export to-hf --io-workers 8          # Threads for S3 download + crop
uv run pipeline.py export to-hf --sample                # Upload a sample only
```

Each chunk writes its own parquet shards and uploads its own images. Combine shards afterward with a final upload step.

</details>

<details>
<summary><h3>export viewer-space</h3></summary>

Export a self-contained HuggingFace Space with static volume data for interactive browsing.

Generates a complete Gradio app directory including:
- `app.py` — generated from `commands/export/templates/viewer_space_app.py`
- `static/` — CSS/HTML assets from `commands/export/templates/static/`
- Pre-exported volume images (compressed JPEG, max 1400px) and JSON metadata
- `requirements.txt` and Space `README.md`

The generated Space shows bounding box annotations, classifications, and captions for a set of demo volumes defined in `VOLUME_BARCODES`.

```bash
uv run pipeline.py export viewer-space
uv run pipeline.py export viewer-space --output-dir ./my-space
uv run pipeline.py export viewer-space --detections-only  # Only pages with detections
uv run pipeline.py export viewer-space --push             # Push to HuggingFace Space
```

When `--push` is used, images are synced to a HF data bucket and the Space files are uploaded to a HF Space repo.

</details>

<details>
<summary><h3>export peek</h3></summary>

Peek at random samples to visually confirm the pipeline is working as expected.

Supports both batch-level steps (detection, classification, captioning) and run-level steps (embedding, hash deduplication).

```bash
uv run pipeline.py export peek --scope deduplication --id-pipeline-batch 1 --n 10
uv run pipeline.py export peek --scope detection --id-pipeline-batch 1 --n all
uv run pipeline.py export peek --scope detection --id-pipeline-batch 1 --n 5 --output-dir peek-5-volumes/
```

</details>

<details>
<summary><h3>export stats</h3></summary>

Generate aggregate statistics and visualizations from the database.

Creates PNG charts and a JSON summary covering table counts, classification distributions, confidence scores, crop dimensions, caption statistics, deduplication effectiveness, pipeline performance, and volume metadata.

```bash
uv run pipeline.py export stats
uv run pipeline.py export stats --output-dir ./my_stats
uv run pipeline.py export stats --skip-slow  # Skip expensive queries
```

</details>

[👆 Back to the summary](#summary)

---

## CLI: post-processing

> Post-processing commands operate on pipeline data after a run is complete. Some require the `filtered_dataset` view.

<details>
<summary><h3>post-processing backfill</h3></summary>

Backfill computed columns on the `caption` table. Computes and stores:
- `lang_detected`: ISO 639-3 language code via lingua language detection
- `linear_prob`: Geometric mean of token probabilities from OpenAI logprobs
- `thesaurus_matches`: ChronAm thesaurus term matches (JSONB) — **optional**, requires `HF_TOKEN`

Uses a process pool (default 4 workers) because lingua holds the GIL. Each worker loads its own lingua model (~200MB each). Runs periodic VACUUM on the caption table.

NOTE: Run this before exports that need `lang_detected`, `linear_prob`, or `thesaurus_matches` columns. The thesaurus step is optional and can be skipped with `--skip-thesaurus`.

```bash
uv run pipeline.py post-processing backfill
uv run pipeline.py post-processing backfill --force              # Re-compute already-backfilled captions
uv run pipeline.py post-processing backfill --limit 1000         # Process only 1000 captions (testing)
uv run pipeline.py post-processing backfill --cpus-limit 8       # Use 8 worker processes
uv run pipeline.py post-processing backfill --skip-thesaurus     # Skip ChronAm thesaurus
```

</details>

<details>
<summary><h3>post-processing count-tokens</h3></summary>

Count tokens and compute corpus statistics for the `caption_text` column in `filtered_dataset` using tiktoken.

Outputs statistics including total tokens, mean/median/std/percentile tokens per document. Writes results to a JSON file.

```bash
uv run pipeline.py post-processing count-tokens
uv run pipeline.py post-processing count-tokens --encoding o200k_base   # Default encoding
uv run pipeline.py post-processing count-tokens --output logs/token_stats.json
uv run pipeline.py post-processing count-tokens --workers 8
```

</details>

<details>
<summary><h3>post-processing embedding-atlas</h3></summary>

Create an interactive 2D embedding visualization using [Apple's embedding-atlas](https://github.com/apple/embedding-atlas).

Samples records from `filtered_dataset` that have embeddings, prepares a parquet file, and launches an interactive server (or exports as standalone HTML).

```bash
uv run pipeline.py post-processing embedding-atlas --sample 10000
uv run pipeline.py post-processing embedding-atlas --sample 50000 --export-html atlas.html
uv run pipeline.py post-processing embedding-atlas --port 8080 --host 0.0.0.0
uv run pipeline.py post-processing embedding-atlas --text-column caption_text
uv run pipeline.py post-processing embedding-atlas --no-serve  # Prepare data only
```

</details>

[👆 Back to the summary](#summary)

---

## About IDI
The Institutional Data Initiative at Harvard Law School Library works with knowledge institutions—from libraries and museums to cultural groups and government agencies—to refine and publish their collections as data. [Reach out to collaborate on your collections](https://institutionaldatainitiative.org/#get-involved).

---

## Cite

> TODO

```bibtext
```

[👆 Back to the summary](#summary)
