# 📚 Institutional Books Visual Elements Pipeline
The Institutional Data Initiative's pipeline for extracting, analyzing, and publishing visual elements from the Institutional Books 1.0 collection.

> 🚧 Work in progress, experimental

- 🤗 [Institutional Books on HuggingFace](https://huggingface.co/collections/instdin/institutional-books-68366258bfb38364238477cf)
- 📄 [Technical report](TODO)
- 🌐 [Website](TODO)

---

## Summary 
- [Getting started](#getting-started)
- [Concept](#concept)
- [Sequencing of a pipeline run](#sequencing-of-a-pipeline-run)
- [Available utilities](#available-utilities)
- [Custom exclusion List](#custom-exclusion-list)
- [CLI: `system`](#cli-system)
- [CLI: `orchestration`](#cli-system)
- [CLI: `steps`](#cli-steps)
- [CLI: `export`](#cli-export)
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
  - pgvector extension
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

**NOTE:** 
- All commands come with a `--help` flag to list options.
- We recommend storing logs individually, as such: 
```bash
uv run pipeline.py --verbose command options &> run-{date}-{info}.log
```

[👆 Back to the summary](#summary)

---

## Concept

> Work in progress

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

> Work in progress

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

### 4. Export

"Peek" at samples to confirm the pipeline is working as expected:
> TODO

Intermediary data for analysis:
> TODO

Full dataset:
> TODO

[👆 Back to the summary](#summary)

---

## Available utilities

> Work in progress

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

> Work in progress

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

> Work in progress

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

> Work in progress

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


[👆 Back to the summary](#summary)

---

## CLI: export

> Work in progress

<details>
<summary><h3>export peek 🚧</h3></summary>
</details>

<details>
<summary><h3>export as-jsonl 🚧</h3></summary>
</details>

<details>
<summary><h3>export to-hf 🚧</h3></summary>
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
