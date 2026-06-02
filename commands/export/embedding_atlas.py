"""
Embedding Atlas - Create interactive visualization of image embeddings.

This tool creates an interactive 2D visualization of the filtered dataset embeddings
using Apple's embedding-atlas library, sampling directly from the database.
"""

import click
from loguru import logger
from pathlib import Path
import json
import subprocess

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils import get_db
from const import ANALYSIS_OUTPUT_DIR, DATETIME_SLUG


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


def fetch_filtered_dataset_sample(sample_size: int) -> pd.DataFrame:
    """
    Fetch a random sample from the filtered_dataset view in the database.
    Only fetches records that have embeddings.

    Samples by randomly picking IDs from the pipeline_batch_item table
    (a real indexed table), then fetching their rows from the view.
    """
    import random

    conn = _get_raw_connection()
    try:
        cur = conn.cursor()

        # Fast: grab all item IDs from the indexed base table
        cur.execute("SELECT id_pipeline_batch_item FROM pipeline_batch_item")
        item_ids = [row[0] for row in cur.fetchall()]
        cur.close()
    finally:
        try:
            conn.rollback()
        except Exception:
            pass

    if not item_ids:
        return pd.DataFrame()

    logger.info(f"Total items in pipeline_batch_item: {len(item_ids):,}")

    if sample_size:
        items_needed = min(len(item_ids), sample_size)
        sampled_ids = random.sample(item_ids, items_needed)
    else:
        sampled_ids = item_ids

    all_rows = []
    col_names = None
    conn = _get_raw_connection()
    batch_size = 200

    try:
        cur = conn.cursor()
        for i in range(0, len(sampled_ids), batch_size):
            batch_ids = sampled_ids[i:i + batch_size]
            cur.execute("""
                SELECT
                    id_detection,
                    barcode,
                    scan_filename,
                    pred_class AS classification,
                    classification_conf AS classification_confidence,
                    caption_text AS caption,
                    caption_lang,
                    bbox_xywh,
                    bbox_conf AS detection_confidence,
                    image_hash,
                    embedding
                FROM filtered_dataset
                WHERE pipeline_batch_item_id = ANY(%s)
                  AND embedding IS NOT NULL
            """, (batch_ids,))
            rows = cur.fetchall()
            if col_names is None and rows:
                col_names = [desc[0] for desc in cur.description]
            all_rows.extend(rows)

            if sample_size and len(all_rows) >= sample_size:
                break

        cur.close()
    finally:
        try:
            conn.rollback()
        except Exception:
            pass

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=col_names)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    return df


def _parse_embedding(val):
    """Parse an embedding value from PostgreSQL into a list of floats."""
    if val is None:
        return None
    if isinstance(val, list):
        return val if len(val) > 0 else None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        # PostgreSQL array format: {0.1,0.2,...}
        if val.startswith("{") and val.endswith("}"):
            return [float(x) for x in val[1:-1].split(",") if x]
        # JSON format: [0.1, 0.2, ...]
        if val.startswith("["):
            return json.loads(val)
        return [float(x) for x in val.split(",") if x]
    # Iterable (e.g. psycopg2 array)
    try:
        return [float(x) for x in val]
    except (TypeError, ValueError):
        return None


def prepare_atlas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for embedding-atlas.
    Parses embeddings and computes width/height from bbox_xywh.
    """
    logger.info(f"Preparing data for atlas ({len(df)} records)...")

    if "embedding" in df.columns:
        df["embedding"] = df["embedding"].apply(_parse_embedding)

    has_embedding = df["embedding"].apply(lambda x: x is not None)
    original_count = len(df)
    df = df[has_embedding].copy()
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} records without valid embeddings")

    if "bbox_xywh" in df.columns:
        df["width"] = df["bbox_xywh"].apply(
            lambda x: int(round(x[2])) if x and len(x) >= 4 else None
        )
        df["height"] = df["bbox_xywh"].apply(
            lambda x: int(round(x[3])) if x and len(x) >= 4 else None
        )
        df["pixel_count_mpx"] = df.apply(
            lambda row: (row["width"] * row["height"]) / 1_000_000
            if row["width"] and row["height"] else None,
            axis=1,
        )
        df = df.drop(columns=["bbox_xywh"])

    logger.info(f"Prepared {len(df)} records for visualization")
    return df


def create_atlas_parquet(df: pd.DataFrame, output_path: Path, text_column: str) -> Path:
    """
    Create a parquet file formatted for embedding-atlas CLI.
    """
    logger.info("Creating atlas-compatible parquet file...")

    atlas_columns = [
        "barcode",
        "scan_filename",
        "classification",
        "classification_confidence",
        text_column,
        "caption_lang",
        "width",
        "height",
        "pixel_count_mpx",
        "detection_confidence",
        "image_hash",
        "embedding",
    ]

    existing_cols = [c for c in atlas_columns if c in df.columns]
    atlas_df = df[existing_cols].copy()

    if "embedding" in atlas_df.columns:
        atlas_df["embedding"] = atlas_df["embedding"].apply(
            lambda x: list(x) if x is not None else None
        )

    float_cols = ["classification_confidence", "detection_confidence", "pixel_count_mpx"]
    for col in float_cols:
        if col in atlas_df.columns:
            atlas_df[col] = pd.to_numeric(atlas_df[col], errors="coerce")

    int_cols = ["width", "height"]
    for col in int_cols:
        if col in atlas_df.columns:
            atlas_df[col] = pd.to_numeric(atlas_df[col], errors="coerce").astype("Int64")

    str_cols = ["barcode", "scan_filename", "classification", "caption_lang", "image_hash", text_column]
    for col in str_cols:
        if col in atlas_df.columns:
            atlas_df[col] = atlas_df[col].apply(lambda x: str(x) if x is not None else None)

    output_file = output_path / "atlas_data.parquet"

    # Build embedding column as a typed list array; use pandas for the rest
    embedding_col = pa.array(atlas_df.pop("embedding").tolist(), type=pa.list_(pa.float32()))
    rest_table = pa.Table.from_pandas(atlas_df, preserve_index=False)
    table = rest_table.append_column("embedding", embedding_col)
    pq.write_table(table, output_file)

    logger.info(f"Atlas parquet saved to: {output_file}")
    return output_file


@click.command("embedding-atlas")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=ANALYSIS_OUTPUT_DIR,
    help="Output directory for atlas files",
)
@click.option(
    "--text-column",
    type=str,
    default="caption",
    help="Column to use for text display (default: caption)",
)
@click.option(
    "--export-html",
    type=click.Path(),
    default=None,
    help="Export as standalone HTML application to this path",
)
@click.option(
    "--port",
    type=int,
    default=5055,
    help="Port for the embedding-atlas server (default: 5055)",
)
@click.option(
    "--host",
    type=str,
    default="localhost",
    help="Host for the embedding-atlas server (default: localhost)",
)
@click.option(
    "--sample",
    type=int,
    default=10000,
    help="Random sample size from the database (default: 10000)",
)
@click.option(
    "--no-serve",
    is_flag=True,
    help="Only prepare data, don't launch the server",
)
@click.option(
    "--cors",
    is_flag=True,
    help="Enable CORS for cross-origin requests",
)
def embedding_atlas(
    output_dir,
    text_column,
    export_html,
    port,
    host,
    sample,
    no_serve,
    cors,
):
    """
    Create an interactive embedding atlas visualization.

    Uses Apple's embedding-atlas to visualize image embeddings from the
    filtered_dataset view in the database. The visualization shows a 2D
    projection of the high-dimensional embeddings with interactive exploration.

    Examples:
        embedding-atlas --sample 5000
        embedding-atlas --sample 20000 --export-html atlas.html
        embedding-atlas --port 8080 --host 0.0.0.0
    """
    output_path = Path(output_dir) / f"embedding_atlas_{DATETIME_SLUG}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching sample of {sample} records from filtered_dataset...")
    df = fetch_filtered_dataset_sample(sample)
    logger.info(f"Fetched {len(df)} records from database")

    df = prepare_atlas_data(df)

    if len(df) == 0:
        logger.error("No records with embeddings found in filtered_dataset")
        return

    atlas_file = create_atlas_parquet(df, output_path, text_column)

    if no_serve:
        logger.success(f"Atlas data prepared: {atlas_file}")
        logger.info("To visualize, run:")
        logger.info(f"  embedding-atlas {atlas_file} --vector embedding --text {text_column}")
        return

    cmd = [
        "embedding-atlas",
        str(atlas_file),
        "--vector", "embedding",
        "--text", text_column,
        "--port", str(port),
        "--host", host,
    ]

    if export_html:
        export_path = Path(export_html)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--export-application", str(export_path)])
        logger.info(f"Will export to: {export_path}")

    if cors:
        cmd.append("--cors")

    logger.info(f"Launching embedding-atlas...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            ["embedding-atlas", "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        logger.error("embedding-atlas is not installed.")
        logger.info("Install it with: pip install embedding-atlas")
        logger.info("Or: uv add embedding-atlas")
        logger.info(f"\nData file prepared at: {atlas_file}")
        return

    try:
        logger.info(f"Starting server at http://{host}:{port}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"embedding-atlas failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        logger.info("Server stopped")

    logger.success(f"Atlas files saved to: {output_path}")
