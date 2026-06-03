import click
from loguru import logger
import json
import math
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from lingua import LanguageDetectorBuilder

from models import Caption
from utils import get_db, process_db_write_batch
from const import DEFAULT_DB_BATCH_SIZE, BACKFILL_DEFAULT_WORKERS, VACUUM_EVERY_N_CHUNKS, HF_THESAURUS_REPO 


def load_thesaurus() -> tuple[dict[str, str], re.Pattern | None]:
    """
    Load thesaurus from HuggingFace and return (term_to_category, compiled_regex).
    The regex matches any term in a single pass.
    """
    import os
    from datasets import load_dataset

    term_to_category = {}

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set, cannot load thesaurus from HuggingFace")
        return term_to_category, None

    try:
        dataset = load_dataset(HF_THESAURUS_REPO, split="train", token=hf_token)
    except Exception as e:
        logger.warning(f"Failed to load thesaurus from {HF_THESAURUS_REPO}: {e}")
        return term_to_category, None

    for entry in dataset:
        category = entry.get("category", "unknown")
        if entry.get("keyword"):
            term_to_category[entry["keyword"].lower()] = category
        for term in entry.get("related_terms", []):
            if term:
                term_to_category[term.lower()] = category

    if not term_to_category:
        return term_to_category, None

    sorted_terms = sorted(term_to_category.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b"
    compiled = re.compile(pattern)

    return term_to_category, compiled


def get_thesaurus_matches(
    caption_text: str,
    term_to_category: dict[str, str],
    compiled_regex: re.Pattern,
) -> dict | None:
    if not caption_text:
        return None

    caption_lower = caption_text.lower()
    found_terms = compiled_regex.findall(caption_lower)
    if not found_terms:
        return None

    matches: dict[str, dict[str, int]] = {}
    for term in found_terms:
        category = term_to_category[term]
        if category not in matches:
            matches[category] = {}
        matches[category][term] = matches[category].get(term, 0) + 1

    return matches


def detect_caption_language(text: str, detector) -> str | None:
    if not text or not text.strip():
        return None
    try:
        detected = detector.detect_language_of(text)
        if detected:
            return detected.iso_code_639_3.name.lower()
    except Exception:
        pass
    return None


def calculate_caption_linear_prob(logprobs_data) -> float | None:
    if not logprobs_data or not isinstance(logprobs_data, list):
        return None

    logprob_values = []
    for item in logprobs_data:
        if isinstance(item, dict) and "logprob" in item:
            lp = item["logprob"]
            if lp is not None and isinstance(lp, (int, float)):
                logprob_values.append(lp)

    if not logprob_values:
        return None

    mean_logprob = sum(logprob_values) / len(logprob_values)
    return math.exp(mean_logprob)


def ensure_backfill_columns():
    """Add backfill columns to the caption table if they don't exist yet."""
    db = get_db()
    migrations = [
        "ALTER TABLE caption ADD COLUMN IF NOT EXISTS lang_detected VARCHAR(10)",
        "ALTER TABLE caption ADD COLUMN IF NOT EXISTS linear_prob DOUBLE PRECISION",
        "ALTER TABLE caption ADD COLUMN IF NOT EXISTS thesaurus_matches JSONB",
    ]
    for sql in migrations:
        db.execute_sql(sql)


_worker_detector = None
_worker_thesaurus = None
_worker_regex = None
_worker_skip_thesaurus = False


def _init_worker(skip_thesaurus: bool = False):
    """Called once per worker process to initialize DB and heavy resources."""
    global _worker_detector, _worker_thesaurus, _worker_regex, _worker_skip_thesaurus
    _worker_skip_thesaurus = skip_thesaurus
    get_db()
    if not skip_thesaurus:
        _worker_thesaurus, _worker_regex = load_thesaurus()
    _worker_detector = LanguageDetectorBuilder.from_all_languages().build()


def backfill_worker(caption_ids: list[int], batch_size: int) -> int:
    """
    Process a single chunk of caption IDs. Returns the number processed.
    The lingua detector and thesaurus are initialized once per process via _init_worker.
    """
    captions = list(Caption.select().where(Caption.id_caption << caption_ids))
    count = len(captions)

    for cap in captions:
        text = cap.text if cap.text else None

        cap.lang_detected = detect_caption_language(text, _worker_detector)
        cap.linear_prob = calculate_caption_linear_prob(cap.logprobs)
        if not _worker_skip_thesaurus:
            cap.thesaurus_matches = (
                get_thesaurus_matches(text, _worker_thesaurus, _worker_regex)
                if _worker_regex
                else None
            )

    fields_to_update = [Caption.lang_detected, Caption.linear_prob]
    if not _worker_skip_thesaurus:
        fields_to_update.append(Caption.thesaurus_matches)

    process_db_write_batch(
        Caption,
        entries_to_update=captions,
        fields_to_update=fields_to_update,
    )
    return count


@click.command("backfill")
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_DB_BATCH_SIZE,
    help="Number of captions to process per DB write batch",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-compute even for captions that already have backfilled values",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit the number of captions to process (for testing)",
)
@click.option(
    "--cpus-limit",
    type=int,
    default=BACKFILL_DEFAULT_WORKERS,
    help=f"Number of parallel worker processes (default: {BACKFILL_DEFAULT_WORKERS})",
)
@click.option(
    "--skip-thesaurus",
    is_flag=True,
    help="Skip ChronAm thesaurus matching (avoids HF_TOKEN requirement)",
)
def backfill(batch_size, force, limit, cpus_limit, skip_thesaurus):
    """
    Backfill computed columns on the caption table.

    Computes and stores:
    - lang_detected: ISO 639-3 code from lingua language detection
    - linear_prob: geometric mean of token probabilities from logprobs
    - thesaurus_matches: ChronAm thesaurus term matches (JSONB) [optional]

    Uses a process pool (default 4 workers) because lingua holds the GIL.
    Each worker loads its own lingua model (~200MB each). Runs periodic VACUUM
    on the caption table.

    NOTE: Run this before exports that need lang_detected, linear_prob, or
    thesaurus_matches columns. The thesaurus step is optional and can be skipped
    with --skip-thesaurus.

    Examples:
        backfill
        backfill --force
        backfill --limit 1000
        backfill --cpus-limit 8
        backfill --skip-thesaurus
    """
    logger.info("Starting caption backfill...")
    if skip_thesaurus:
        logger.info("  Skipping ChronAm thesaurus matching (--skip-thesaurus)")

    ensure_backfill_columns()

    query = Caption.select(Caption.id_caption)
    if not force:
        query = query.where(Caption.lang_detected.is_null())
    if limit:
        query = query.limit(limit)

    caption_ids = [cap.id_caption for cap in query.iterator()]
    total = len(caption_ids)
    logger.info(f"  Captions to process: {total}")

    if total == 0:
        logger.info("Nothing to backfill.")
        return

    processes_total = min(cpus_limit, total)
    logger.info(f"  Launching {processes_total} worker processes...")

    # Split IDs into batch_size chunks submitted as individual tasks.
    # The pool acts as a work queue: fast workers pull the next chunk automatically,
    # avoiding straggler imbalance from static round-robin assignment.
    chunks = [caption_ids[i : i + batch_size] for i in range(0, total, batch_size)]
    logger.info(f"  Submitting {len(chunks)} chunks of up to {batch_size} captions")

    processed = 0
    chunks_completed = 0

    with ProcessPoolExecutor(max_workers=processes_total, initializer=_init_worker, initargs=(skip_thesaurus,)) as executor:
        futures = {
            executor.submit(backfill_worker, caption_ids=chunk, batch_size=batch_size): idx
            for idx, chunk in enumerate(chunks)
        }

        for fut in as_completed(futures):
            try:
                processed += fut.result()
            except Exception:
                logger.error("Error in a worker process:\n" + traceback.format_exc())
                executor.shutdown(wait=False, cancel_futures=True)
                click.get_current_context().exit(1)

            chunks_completed += 1
            logger.info(f"  Progress: {processed}/{total} captions")

            if chunks_completed % BACKFILL_VACUUM_EVERY_N_CHUNKS == 0:
                _run_vacuum()

    _run_vacuum()
    logger.success(f"Backfill complete. Processed {processed} captions across {processes_total} processes.")


def _run_vacuum():
    """Run VACUUM on the caption table with a fresh connection to avoid stale SSL errors."""
    try:
        import psycopg2, os
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            sslmode=os.getenv("POSTGRES_SSLMODE", "require"),
        )
        conn.autocommit = True
        logger.info("  Running VACUUM on caption table...")
        conn.cursor().execute("VACUUM caption")
        conn.close()
        logger.info("  VACUUM complete.")
    except Exception as e:
        logger.warning(f"  VACUUM failed (non-fatal): {e}")
