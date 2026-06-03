import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
import click
import tiktoken
from tqdm import tqdm
from loguru import logger
from utils import get_db
from const import COUNT_TOKENS_SERVER_SIDE_CURSOR_SIZE, COUNT_TOKENS_WORKERS


def _get_raw_connection():
    db = get_db()
    conn = db.connection()
    if conn.autocommit:
        conn.autocommit = False
    else:
        conn.rollback()
    return conn


@click.command("count-tokens")
@click.option(
    "--encoding",
    "encoding_name",
    type=str,
    default="o200k_base",
    help="Tiktoken encoding to use (default: o200k_base)",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="logs/token_stats.json",
    help="Optional path to write stats as JSON",
)
@click.option(
    "--workers",
    type=int,
    default=COUNT_TOKENS_WORKERS,
    help="Number of threads for tokenization (default: CPU count)",
)
def count_tokens(encoding_name, output_path, workers):
    """Count tokens and compute corpus statistics for the filtered_dataset caption_text column."""
    enc = tiktoken.get_encoding(encoding_name)
    num_workers = workers

    conn = _get_raw_connection()

    logger.info("Counting rows in filtered_dataset...")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM filtered_dataset")
    total_count = cur.fetchone()[0]
    cur.close()
    logger.info(f"Total rows: {total_count:,}")

    token_counts = []
    empty_rows = 0

    logger.info(f"Using {num_workers} workers for tokenization")

    def _encode(text):
        return len(enc.encode(text))

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        with conn.cursor(name="count_tokens_cursor") as cursor:
            cursor.itersize = COUNT_TOKENS_SERVER_SIDE_CURSOR_SIZE
            cursor.execute("SELECT caption_text FROM filtered_dataset")
            with tqdm(total=total_count, unit="rows") as pbar:
                while True:
                    rows = cursor.fetchmany(COUNT_TOKENS_SERVER_SIDE_CURSOR_SIZE)
                    if not rows:
                        break
                    texts = [text for (text,) in rows if text]
                    empty_rows += len(rows) - len(texts)
                    counts = list(pool.map(_encode, texts))
                    token_counts.extend(counts)
                    pbar.update(len(rows))
                    pbar.set_postfix(tokens=f"{sum(token_counts):,}")

    total_tokens = sum(token_counts)
    total_docs = len(token_counts)
    sorted_counts = sorted(token_counts)

    def percentile(sorted_data, p):
        k = (len(sorted_data) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    stats = {
        "encoding": encoding_name,
        "total_documents": total_docs,
        "empty_documents": empty_rows,
        "total_tokens": total_tokens,
        "mean_tokens_per_document": total_tokens / total_docs if total_docs else 0,
        "median_tokens_per_document": statistics.median(token_counts) if token_counts else 0,
        "std_tokens_per_document": statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
        "min_tokens": sorted_counts[0] if sorted_counts else 0,
        "max_tokens": sorted_counts[-1] if sorted_counts else 0,
        "p5": percentile(sorted_counts, 5) if sorted_counts else 0,
        "p25": percentile(sorted_counts, 25) if sorted_counts else 0,
        "p75": percentile(sorted_counts, 75) if sorted_counts else 0,
        "p95": percentile(sorted_counts, 95) if sorted_counts else 0,
        "p99": percentile(sorted_counts, 99) if sorted_counts else 0,
    }

    logger.info("--- Corpus Token Statistics ---")
    logger.info(f"Encoding: {stats['encoding']}")
    logger.info(f"Total documents: {stats['total_documents']:,}")
    logger.info(f"Empty documents: {stats['empty_documents']:,}")
    logger.info(f"Total tokens: {stats['total_tokens']:,}")
    logger.info(f"Mean tokens/doc: {stats['mean_tokens_per_document']:.1f}")
    logger.info(f"Median tokens/doc: {stats['median_tokens_per_document']:.1f}")
    logger.info(f"Std tokens/doc: {stats['std_tokens_per_document']:.1f}")
    logger.info(f"Min tokens: {stats['min_tokens']:,}")
    logger.info(f"Max tokens: {stats['max_tokens']:,}")
    logger.info(f"P5: {stats['p5']:.0f} | P25: {stats['p25']:.0f} | P75: {stats['p75']:.0f} | P95: {stats['p95']:.0f} | P99: {stats['p99']:.0f}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats written to {output_path}")
