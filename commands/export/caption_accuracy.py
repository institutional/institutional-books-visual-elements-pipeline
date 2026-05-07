"""
Caption Accuracy - Compare captions with images using multi-modal embeddings.

This tool evaluates caption quality by computing similarity scores between
images and their captions using CLIP (or similar multi-modal embedding models).
"""

import click
from loguru import logger
from pathlib import Path
import json
import base64
import html as html_module

import numpy as np
import pandas as pd
import torch
from PIL import Image
import io

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


def load_clip_model(model_name: str, device: str):
    """
    Load CLIP model and processor.
    """
    from transformers import CLIPProcessor, CLIPModel

    logger.info(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on device: {device}")
    return model, processor


def extract_image_from_volume(barcode: str, scan_filename: str, bbox_xyxy: list) -> bytes | None:
    """
    Extract a cropped image from a volume using the pipeline's data loading mechanism.
    Returns image bytes or None if not found.
    """
    from models import PipelineBatchItem, IBVolume
    import cv2
    import numpy as np

    try:
        # Find the volume
        volume = IBVolume.get_or_none(IBVolume.barcode == barcode)
        if not volume:
            return None

        # Find a pipeline batch item for this volume
        item = (
            PipelineBatchItem.select()
            .where(PipelineBatchItem.ib_volume == volume)
            .first()
        )
        if not item:
            return None

        # Load the item data
        item_data = item.get_data()
        image_bytes = item_data.images.get(scan_filename)
        if image_bytes is None:
            return None

        # Decode and crop
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        full_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)

        x1, y1, x2, y2 = map(int, bbox_xyxy)
        h, w = full_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped = full_image[y1:y2, x1:x2]

        # Convert BGR to RGB
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Encode as JPEG bytes
        _, encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
        return encoded.tobytes()

    except Exception as e:
        logger.warning(f"Error extracting image for {barcode}/{scan_filename}: {e}")
        return None


def compute_similarity_batch(
    model,
    processor,
    images: list[Image.Image],
    texts: list[str],
    device: str,
) -> np.ndarray:
    """
    Compute cosine similarity between images and their corresponding texts.
    Returns an array of similarity scores.
    """
    with torch.no_grad():
        # Process inputs
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Compute similarity (diagonal elements = image-text pairs)
        similarity = (image_embeds * text_embeds).sum(dim=-1)

        return similarity.cpu().numpy()


def generate_html_viewer(results: list[dict], output_path: Path, stats: dict) -> Path:
    """
    Generate an HTML viewer showing images alongside similarity scores and captions.
    Images are embedded as base64 data URIs.
    """
    # Sort results by similarity (lowest first to highlight potential issues)
    sorted_results = sorted(results, key=lambda x: x["similarity"])

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Caption Accuracy Viewer</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c5282;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }}
        .controls {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .controls label {{
            font-weight: 500;
            color: #333;
        }}
        .controls select, .controls input {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .card-image {{
            width: 100%;
            height: 250px;
            object-fit: contain;
            background: #f0f0f0;
            border-bottom: 1px solid #eee;
        }}
        .card-content {{
            padding: 15px;
        }}
        .similarity-score {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 10px;
        }}
        .score-strong {{
            background: #c6f6d5;
            color: #276749;
        }}
        .score-moderate {{
            background: #fefcbf;
            color: #975a16;
        }}
        .score-weak {{
            background: #fed7d7;
            color: #c53030;
        }}
        .caption {{
            font-size: 14px;
            line-height: 1.5;
            color: #333;
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #4a90a4;
        }}
        .meta {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }}
        .meta span {{
            margin-right: 15px;
        }}
        .classification-badge {{
            display: inline-block;
            padding: 3px 8px;
            background: #e2e8f0;
            border-radius: 4px;
            font-size: 11px;
            color: #4a5568;
        }}
        .hidden {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Caption Accuracy Viewer</h1>
        <p>Sorted by similarity score (lowest first). Model: {html_module.escape(stats.get('model', 'N/A'))}</p>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{stats.get('total_samples', 0)}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('mean_similarity', 0):.3f}</div>
                <div class="stat-label">Mean Similarity</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('median_similarity', 0):.3f}</div>
                <div class="stat-label">Median Similarity</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('quality_buckets', {}).get('strong_match_gt_0.30', 0)}</div>
                <div class="stat-label">Strong (&gt;0.30)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('quality_buckets', {}).get('moderate_match_0.20_0.30', 0)}</div>
                <div class="stat-label">Moderate (0.20-0.30)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('quality_buckets', {}).get('weak_match_lt_0.20', 0)}</div>
                <div class="stat-label">Weak (&lt;0.20)</div>
            </div>
        </div>
    </div>

    <div class="controls">
        <div>
            <label for="sort-select">Sort by: </label>
            <select id="sort-select" onchange="sortCards()">
                <option value="similarity-asc">Similarity (Low to High)</option>
                <option value="similarity-desc">Similarity (High to Low)</option>
            </select>
        </div>
        <div>
            <label for="filter-select">Filter: </label>
            <select id="filter-select" onchange="filterCards()">
                <option value="all">All</option>
                <option value="weak">Weak (&lt;0.20)</option>
                <option value="moderate">Moderate (0.20-0.30)</option>
                <option value="strong">Strong (&gt;0.30)</option>
            </select>
        </div>
        <div>
            <label for="class-select">Classification: </label>
            <select id="class-select" onchange="filterCards()">
                <option value="all">All</option>
            </select>
        </div>
        <div>
            <label for="search-input">Search caption: </label>
            <input type="text" id="search-input" placeholder="Type to search..." oninput="filterCards()">
        </div>
    </div>

    <div class="grid" id="cards-grid">
'''

    # Track unique classifications for the filter
    classifications = set()

    for i, result in enumerate(sorted_results):
        similarity = result["similarity"]
        caption = html_module.escape(str(result.get("caption", "")))
        barcode = html_module.escape(str(result.get("barcode", "")))
        scan_filename = html_module.escape(str(result.get("scan_filename", "")))
        classification = str(result.get("classification") or "Unknown")
        classifications.add(classification)
        classification_escaped = html_module.escape(classification)

        # Determine score class
        if similarity > 0.30:
            score_class = "score-strong"
            score_category = "strong"
        elif similarity >= 0.20:
            score_class = "score-moderate"
            score_category = "moderate"
        else:
            score_class = "score-weak"
            score_category = "weak"

        # Get base64 image if available
        image_b64 = result.get("image_b64", "")
        if image_b64:
            img_src = f"data:image/jpeg;base64,{image_b64}"
        else:
            img_src = ""

        html_content += f'''
        <div class="card" data-similarity="{similarity}" data-category="{score_category}" data-classification="{classification_escaped}">
            {"<img class='card-image' src='" + img_src + "' alt='Detection image' loading='lazy'>" if img_src else "<div class='card-image' style='display:flex;align-items:center;justify-content:center;color:#999;'>Image not available</div>"}
            <div class="card-content">
                <span class="similarity-score {score_class}">{similarity:.4f}</span>
                <span class="classification-badge">{classification_escaped}</span>
                <div class="caption">{caption}</div>
                <div class="meta">
                    <span><strong>Barcode:</strong> {barcode}</span>
                    <span><strong>Scan:</strong> {scan_filename}</span>
                </div>
            </div>
        </div>
'''

    # Add classification options to the filter
    class_options = "".join(
        f'<option value="{html_module.escape(cls)}">{html_module.escape(cls)}</option>'
        for cls in sorted(classifications)
    )

    html_content += f'''
    </div>

    <script>
        // Add classification options
        document.getElementById('class-select').innerHTML += `{class_options}`;

        function sortCards() {{
            const grid = document.getElementById('cards-grid');
            const cards = Array.from(grid.querySelectorAll('.card'));
            const sortValue = document.getElementById('sort-select').value;

            cards.sort((a, b) => {{
                const simA = parseFloat(a.dataset.similarity);
                const simB = parseFloat(b.dataset.similarity);
                if (sortValue === 'similarity-asc') {{
                    return simA - simB;
                }} else {{
                    return simB - simA;
                }}
            }});

            cards.forEach(card => grid.appendChild(card));
        }}

        function filterCards() {{
            const filterValue = document.getElementById('filter-select').value;
            const classValue = document.getElementById('class-select').value;
            const searchValue = document.getElementById('search-input').value.toLowerCase();
            const cards = document.querySelectorAll('.card');

            cards.forEach(card => {{
                const category = card.dataset.category;
                const classification = card.dataset.classification;
                const caption = card.querySelector('.caption').textContent.toLowerCase();

                let show = true;

                // Filter by category
                if (filterValue !== 'all' && category !== filterValue) {{
                    show = false;
                }}

                // Filter by classification
                if (classValue !== 'all' && classification !== classValue) {{
                    show = false;
                }}

                // Filter by search
                if (searchValue && !caption.includes(searchValue)) {{
                    show = false;
                }}

                card.classList.toggle('hidden', !show);
            }});
        }}
    </script>
</body>
</html>
'''

    html_file = output_path / "viewer.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_file


@click.command("caption-accuracy")
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
    help="Output directory for results",
)
@click.option(
    "--sample",
    type=int,
    default=1000,
    help="Number of random samples to evaluate (default: 1000)",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for model inference (default: 32)",
)
@click.option(
    "--model",
    "model_name",
    type=str,
    default="openai/clip-vit-base-patch32",
    help="CLIP model to use (default: openai/clip-vit-base-patch32)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (default: auto-detect cuda/cpu)",
)
@click.option(
    "--exclude-failed",
    is_flag=True,
    default=True,
    help="Exclude records with failed captions (default: True)",
)
@click.option(
    "--exclude-na",
    is_flag=True,
    default=True,
    help="Exclude records with NA captions (Ex Libris/Artifact) (default: True)",
)
@click.option(
    "--by-class",
    is_flag=True,
    help="Compute statistics by classification category",
)
@click.option(
    "--export-details",
    is_flag=True,
    help="Export per-sample details to a file",
)
def caption_accuracy(
    input_path,
    output_dir,
    sample,
    batch_size,
    model_name,
    device,
    exclude_failed,
    exclude_na,
    by_class,
    export_details,
):
    """
    Evaluate caption quality using multi-modal embeddings.

    Uses CLIP to compute similarity scores between images and their captions.
    Higher scores indicate better alignment between the image content and caption.

    Typical CLIP similarity scores:
    - > 0.30: Strong match (caption describes the image well)
    - 0.20-0.30: Moderate match (related but not precise)
    - < 0.20: Weak match (may be unrelated or very generic)

    Examples:
        caption-accuracy --input data/cache/temp_analysis/filtered_dataset_xxx
        caption-accuracy --input filtered.parquet --sample 500
        caption-accuracy --input filtered.jsonl --by-class --export-details
        caption-accuracy --input filtered.parquet --model openai/clip-vit-large-patch14
    """
    input_path = Path(input_path)
    output_path = Path(output_dir) / f"caption_accuracy_{DATETIME_SLUG}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load the dataset
    logger.info(f"Loading dataset from {input_path}...")
    df = load_filtered_dataset(input_path)
    logger.info(f"Loaded {len(df)} records")

    # Filter out records without valid captions
    original_count = len(df)

    if exclude_failed:
        df = df[df["caption"] != "CAPTION FAILED"]
        logger.info(f"Excluded {original_count - len(df)} records with failed captions")

    if exclude_na:
        pre_filter = len(df)
        df = df[df["caption"] != "NA"]
        logger.info(f"Excluded {pre_filter - len(df)} records with NA captions")

    # Filter out empty/null captions
    df = df[df["caption"].notna() & (df["caption"] != "")]
    logger.info(f"Final dataset: {len(df)} records with valid captions")

    if len(df) == 0:
        logger.error("No records with valid captions found")
        return

    # Sample if needed
    if sample and sample < len(df):
        logger.info(f"Sampling {sample} records from {len(df)}...")
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    # Load CLIP model
    model, processor = load_clip_model(model_name, device)

    # Process samples
    logger.info(f"Processing {len(df)} samples...")
    results = []
    failed_loads = 0

    # Parse bbox if it's a JSON string
    def parse_bbox(bbox):
        if isinstance(bbox, str):
            return json.loads(bbox)
        return bbox

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        images = []
        texts = []
        valid_indices = []

        for idx, row in batch_df.iterrows():
            # Extract image
            bbox = parse_bbox(row["detection_bbox_xyxy"])
            image_bytes = extract_image_from_volume(
                row["barcode"],
                row["scan_filename"],
                bbox,
            )

            if image_bytes is None:
                failed_loads += 1
                continue

            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                images.append(image)
                texts.append(row["caption"])
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Error loading image: {e}")
                failed_loads += 1

        if not images:
            continue

        # Compute similarities
        try:
            similarities = compute_similarity_batch(model, processor, images, texts, device)

            for i, idx in enumerate(valid_indices):
                row = df.loc[idx]
                # Convert image to base64 for HTML viewer
                img_buffer = io.BytesIO()
                images[i].save(img_buffer, format="JPEG", quality=85)
                img_b64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

                results.append({
                    "barcode": row["barcode"],
                    "scan_filename": row["scan_filename"],
                    "classification": row.get("classification"),
                    "caption": row["caption"],
                    "similarity": float(similarities[i]),
                    "image_b64": img_b64,
                })
        except Exception as e:
            logger.error(f"Error computing similarities: {e}")

        if (batch_start + batch_size) % 100 == 0 or batch_end == len(df):
            logger.info(f"Processed {batch_end}/{len(df)} samples")

    logger.info(f"Successfully processed {len(results)} samples ({failed_loads} failed to load)")

    if not results:
        logger.error("No results computed")
        return

    # Compute statistics
    results_df = pd.DataFrame(results)
    similarities = results_df["similarity"].values

    stats = {
        "model": model_name,
        "total_samples": len(results),
        "failed_loads": failed_loads,
        "mean_similarity": float(np.mean(similarities)),
        "std_similarity": float(np.std(similarities)),
        "median_similarity": float(np.median(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "percentiles": {
            "p10": float(np.percentile(similarities, 10)),
            "p25": float(np.percentile(similarities, 25)),
            "p50": float(np.percentile(similarities, 50)),
            "p75": float(np.percentile(similarities, 75)),
            "p90": float(np.percentile(similarities, 90)),
        },
        "quality_buckets": {
            "strong_match_gt_0.30": int(np.sum(similarities > 0.30)),
            "moderate_match_0.20_0.30": int(np.sum((similarities >= 0.20) & (similarities <= 0.30))),
            "weak_match_lt_0.20": int(np.sum(similarities < 0.20)),
        },
    }

    # Compute by-class statistics
    if by_class and "classification" in results_df.columns:
        class_stats = {}
        for cls in results_df["classification"].unique():
            if pd.isna(cls):
                continue
            cls_df = results_df[results_df["classification"] == cls]
            cls_similarities = cls_df["similarity"].values
            if len(cls_similarities) > 0:
                class_stats[cls] = {
                    "count": len(cls_similarities),
                    "mean": float(np.mean(cls_similarities)),
                    "std": float(np.std(cls_similarities)),
                    "median": float(np.median(cls_similarities)),
                }
        stats["by_classification"] = class_stats

    # Write stats
    stats_file = output_path / "accuracy_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to: {stats_file}")

    # Generate HTML viewer
    html_file = generate_html_viewer(results, output_path, stats)
    logger.info(f"HTML viewer written to: {html_file}")

    # Export per-sample details if requested (exclude image_b64 to keep file small)
    if export_details:
        details_file = output_path / "accuracy_details.jsonl"
        with open(details_file, "w") as f:
            for result in results:
                # Exclude image_b64 from JSONL export
                result_without_image = {k: v for k, v in result.items() if k != "image_b64"}
                f.write(json.dumps(result_without_image) + "\n")
        logger.info(f"Details written to: {details_file}")

        # Also export sorted by similarity (lowest first for review)
        sorted_results = sorted(results, key=lambda x: x["similarity"])
        low_similarity_file = output_path / "low_similarity_samples.jsonl"
        with open(low_similarity_file, "w") as f:
            for result in sorted_results[:100]:  # Bottom 100
                result_without_image = {k: v for k, v in result.items() if k != "image_b64"}
                f.write(json.dumps(result_without_image) + "\n")
        logger.info(f"Low similarity samples written to: {low_similarity_file}")

    # Print summary
    logger.info("=" * 60)
    logger.info("CAPTION ACCURACY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Samples evaluated: {len(results)}")
    logger.info(f"Failed to load: {failed_loads}")
    logger.info("-" * 60)
    logger.info(f"Mean similarity:   {stats['mean_similarity']:.4f}")
    logger.info(f"Std similarity:    {stats['std_similarity']:.4f}")
    logger.info(f"Median similarity: {stats['median_similarity']:.4f}")
    logger.info(f"Min similarity:    {stats['min_similarity']:.4f}")
    logger.info(f"Max similarity:    {stats['max_similarity']:.4f}")
    logger.info("-" * 60)
    logger.info("Percentiles:")
    for pct, val in stats["percentiles"].items():
        logger.info(f"  {pct}: {val:.4f}")
    logger.info("-" * 60)
    logger.info("Quality buckets:")
    buckets = stats["quality_buckets"]
    total = len(results)
    logger.info(f"  Strong (>0.30):   {buckets['strong_match_gt_0.30']:5d} ({100*buckets['strong_match_gt_0.30']/total:.1f}%)")
    logger.info(f"  Moderate (0.20-0.30): {buckets['moderate_match_0.20_0.30']:5d} ({100*buckets['moderate_match_0.20_0.30']/total:.1f}%)")
    logger.info(f"  Weak (<0.20):     {buckets['weak_match_lt_0.20']:5d} ({100*buckets['weak_match_lt_0.20']/total:.1f}%)")

    if by_class and "by_classification" in stats:
        logger.info("-" * 60)
        logger.info("By classification:")
        for cls, cls_stats in sorted(stats["by_classification"].items(), key=lambda x: -x[1]["mean"]):
            logger.info(f"  {cls}: mean={cls_stats['mean']:.4f}, n={cls_stats['count']}")

    logger.info("=" * 60)
    logger.success(f"Caption accuracy analysis complete! Results: {output_path}")
