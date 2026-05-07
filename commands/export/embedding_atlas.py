"""
Embedding Atlas - Create interactive visualization of image embeddings.

This tool creates an interactive 2D visualization of the filtered dataset embeddings
using Apple's embedding-atlas library.
"""

import click
from loguru import logger
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq

from const import ANALYSIS_OUTPUT_DIR, DATETIME_SLUG


def load_filtered_dataset(input_path: Path) -> pd.DataFrame:
    """
    Load filtered dataset from jsonl or parquet format.
    Returns a pandas DataFrame.
    """
    if input_path.is_dir():
        # Check for parquet files first
        parquet_files = list(input_path.glob("*.parquet"))
        if parquet_files:
            logger.info(f"Loading {len(parquet_files)} parquet file(s)...")
            dfs = []
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

        # Check for jsonl
        jsonl_file = input_path / "filtered_dataset.jsonl"
        if jsonl_file.exists():
            logger.info(f"Loading JSONL from {jsonl_file}...")
            return pd.read_json(jsonl_file, lines=True)

        # Check for json
        json_file = input_path / "filtered_dataset.json"
        if json_file.exists():
            logger.info(f"Loading JSON from {json_file}...")
            return pd.read_json(json_file)

        raise FileNotFoundError(f"No dataset files found in {input_path}")
    else:
        # Single file
        suffix = input_path.suffix.lower()
        if suffix == ".parquet":
            logger.info(f"Loading parquet from {input_path}...")
            return pd.read_parquet(input_path)
        elif suffix == ".jsonl":
            logger.info(f"Loading JSONL from {input_path}...")
            return pd.read_json(input_path, lines=True)
        elif suffix == ".json":
            logger.info(f"Loading JSON from {input_path}...")
            return pd.read_json(input_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


def prepare_atlas_data(df: pd.DataFrame, include_embeddings: bool = True) -> pd.DataFrame:
    """
    Prepare DataFrame for embedding-atlas.
    Handles JSON-serialized columns and filters out records without embeddings.
    """
    logger.info(f"Preparing data for atlas ({len(df)} records)...")

    # Parse JSON-serialized columns if needed
    if "embedding" in df.columns and df["embedding"].dtype == object:
        # Check if it's a JSON string
        first_val = df["embedding"].iloc[0] if len(df) > 0 else None
        if isinstance(first_val, str):
            logger.info("Parsing JSON-serialized embedding column...")
            df["embedding"] = df["embedding"].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x else x
            )

    # Filter out records without embeddings
    if include_embeddings:
        has_embedding = df["embedding"].apply(
            lambda x: x is not None and (isinstance(x, list) and len(x) > 0)
        )
        original_count = len(df)
        df = df[has_embedding].copy()
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} records without embeddings")

    logger.info(f"Prepared {len(df)} records for visualization")
    return df


def create_atlas_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Create a parquet file formatted for embedding-atlas CLI.
    The embedding column needs to be a list/array column.
    """
    import pyarrow as pa

    logger.info("Creating atlas-compatible parquet file...")

    # Select columns for the atlas
    atlas_columns = [
        "barcode",
        "scan_filename",
        "classification",
        "classification_confidence",
        "caption",
        "caption_lang",
        "width",
        "height",
        "pixel_count_mpx",
        "detection_confidence",
        "image_hash",
        "embedding",
    ]

    # Filter to existing columns
    existing_cols = [c for c in atlas_columns if c in df.columns]
    atlas_df = df[existing_cols].copy()

    # Convert embedding to proper format (list of floats)
    if "embedding" in atlas_df.columns:
        atlas_df["embedding"] = atlas_df["embedding"].apply(
            lambda x: list(x) if x is not None else None
        )

    # Write to parquet with proper schema
    output_file = output_path / "atlas_data.parquet"

    # Build the table manually to preserve list type
    arrays = {}
    for col in atlas_df.columns:
        if col == "embedding":
            # Create a list array for embeddings
            embeddings = atlas_df[col].tolist()
            arrays[col] = pa.array(embeddings, type=pa.list_(pa.float32()))
        else:
            arrays[col] = pa.array(atlas_df[col].tolist())

    table = pa.table(arrays)
    pq.write_table(table, output_file)

    logger.info(f"Atlas parquet saved to: {output_file}")
    return output_file


@click.command("embedding-atlas")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to filtered dataset (directory, parquet, jsonl, or json file)",
)
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
    default=None,
    help="Random sample size (for large datasets)",
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
    input_path,
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
    filtered dataset. The visualization shows a 2D projection of the
    high-dimensional embeddings with interactive exploration.

    Examples:
        embedding-atlas --input data/cache/temp_analysis/filtered_dataset_xxx
        embedding-atlas --input filtered.parquet --sample 10000
        embedding-atlas --input filtered.jsonl --export-html atlas.html
        embedding-atlas --input filtered.parquet --port 8080 --host 0.0.0.0
    """
    input_path = Path(input_path)
    output_path = Path(output_dir) / f"embedding_atlas_{DATETIME_SLUG}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the dataset
    logger.info(f"Loading dataset from {input_path}...")
    df = load_filtered_dataset(input_path)
    logger.info(f"Loaded {len(df)} records")

    # Sample if requested
    if sample and sample < len(df):
        logger.info(f"Sampling {sample} records from {len(df)}...")
        df = df.sample(n=sample, random_state=42)

    # Prepare data for atlas
    df = prepare_atlas_data(df)

    if len(df) == 0:
        logger.error("No records with embeddings found in dataset")
        return

    # Create atlas-compatible parquet
    atlas_file = create_atlas_parquet(df, output_path)

    if no_serve:
        logger.success(f"Atlas data prepared: {atlas_file}")
        logger.info("To visualize, run:")
        logger.info(f"  embedding-atlas {atlas_file} --vector embedding --text {text_column}")
        return

    # Build the embedding-atlas command
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

    # Check if embedding-atlas is installed
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

    # Run embedding-atlas
    try:
        logger.info(f"Starting server at http://{host}:{port}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"embedding-atlas failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        logger.info("Server stopped")

    logger.success(f"Atlas files saved to: {output_path}")
