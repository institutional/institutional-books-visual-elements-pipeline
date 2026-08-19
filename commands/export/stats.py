import json
from pathlib import Path
from collections import Counter, defaultdict

import click
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tiktoken
from loguru import logger

from utils import get_db
from models import (
    PipelineRun,
    PipelineBatch,
    PipelineBatchItem,
    IBVolume,
    Detection,
    Classification,
    Caption,
    ImageHash,
    ImageEmbedding,
    DedupedHash,
    DedupedEmbedding,
)
from const import CLASSIFICATION_CLASS_DICT, ANALYSIS_OUTPUT_DIR, DATETIME_SLUG

# Use non-interactive backend for server environments
matplotlib.use("Agg")

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE = (12, 6)
FIGSIZE_LARGE = (14, 8)
COLORS = plt.cm.Set3.colors


def get_table_counts():
    """Get row counts for all tables."""
    counts = {
        "pipeline_runs": PipelineRun.select().count(),
        "pipeline_batches": PipelineBatch.select().count(),
        "pipeline_batch_items": PipelineBatchItem.select().count(),
        "volumes": IBVolume.select().count(),
        "detections": Detection.select().count(),
        "classifications": Classification.select().count(),
        "captions": Caption.select().count(),
        "image_hashes": ImageHash.select().count(),
        "image_embeddings": ImageEmbedding.select().count(),
        "deduped_hashes": DedupedHash.select().count(),
        "deduped_embeddings": DedupedEmbedding.select().count(),
    }
    return counts


def plot_table_counts(counts: dict, output_path: Path):
    """Bar chart of record counts per table."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    tables = list(counts.keys())
    values = list(counts.values())

    bars = ax.barh(tables, values, color=COLORS[: len(tables)])

    ax.set_xlabel("Number of Records")
    ax.set_title("Database Table Record Counts")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_path / "01_table_counts.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 01_table_counts.png")


def plot_classification_distribution(output_path: Path):
    """Pie chart and bar chart of classification distribution."""
    class_counts = Counter()

    for cls in Classification.select(Classification.pred_class):
        class_counts[cls.pred_class] += 1

    if not class_counts:
        logger.warning("No classifications found")
        return

    # Map class numbers to names
    labels = []
    values = []
    for class_num, count in class_counts.most_common():
        class_name = CLASSIFICATION_CLASS_DICT.get(str(class_num), f"({class_num})")
        labels.append(class_name)
        values.append(count)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=COLORS[: len(labels)],
        pctdistance=0.75,
    )
    ax1.set_title("Classification Distribution (Pie)")

    # Bar chart
    bars = ax2.barh(labels, values, color=COLORS[: len(labels)])
    ax2.set_xlabel("Number of Detections")
    ax2.set_title("Classification Distribution (Bar)")

    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_path / "02_classification_distribution.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 02_classification_distribution.png")


def plot_confidence_distributions(output_path: Path):
    """Histograms of detection and classification confidence scores."""
    # Detection confidence
    det_confs = [
        d.bbox_conf for d in Detection.select(Detection.bbox_conf) if d.bbox_conf is not None
    ]

    # Classification confidence
    cls_confs = [
        c.pred_conf
        for c in Classification.select(Classification.pred_conf)
        if c.pred_conf is not None
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    if det_confs:
        ax1.hist(det_confs, bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
        ax1.set_xlabel("Confidence Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Detection Confidence Distribution\n(n={len(det_confs):,})")
        ax1.axvline(
            np.mean(det_confs), color="red", linestyle="--", label=f"Mean: {np.mean(det_confs):.3f}"
        )
        ax1.axvline(
            np.median(det_confs),
            color="orange",
            linestyle="--",
            label=f"Median: {np.median(det_confs):.3f}",
        )
        ax1.legend()
    else:
        ax1.text(
            0.5,
            0.5,
            "No detection confidence data",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("Detection Confidence Distribution")

    if cls_confs:
        ax2.hist(cls_confs, bins=50, color=COLORS[1], edgecolor="white", alpha=0.8)
        ax2.set_xlabel("Confidence Score")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Classification Confidence Distribution\n(n={len(cls_confs):,})")
        ax2.axvline(
            np.mean(cls_confs), color="red", linestyle="--", label=f"Mean: {np.mean(cls_confs):.3f}"
        )
        ax2.axvline(
            np.median(cls_confs),
            color="orange",
            linestyle="--",
            label=f"Median: {np.median(cls_confs):.3f}",
        )
        ax2.legend()
    else:
        ax2.text(
            0.5,
            0.5,
            "No classification confidence data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Classification Confidence Distribution")

    plt.tight_layout()
    fig.savefig(output_path / "03_confidence_distributions.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 03_confidence_distributions.png")


def plot_crop_dimensions(output_path: Path):
    """Histograms of crop width, height, and area distributions."""
    widths = []
    heights = []
    areas = []

    for det in Detection.select(Detection.bbox_xywh):
        if det.bbox_xywh and len(det.bbox_xywh) == 4:
            w, h = det.bbox_xywh[2], det.bbox_xywh[3]
            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    if not widths:
        logger.warning("No crop dimension data found")
        return

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    # Width histogram
    axes[0, 0].hist(widths, bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
    axes[0, 0].set_xlabel("Width (pixels)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(
        f"Crop Width Distribution\n(mean: {np.mean(widths):.0f}, median: {np.median(widths):.0f})"
    )

    # Height histogram
    axes[0, 1].hist(heights, bins=50, color=COLORS[1], edgecolor="white", alpha=0.8)
    axes[0, 1].set_xlabel("Height (pixels)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(
        f"Crop Height Distribution\n(mean: {np.mean(heights):.0f}, median: {np.median(heights):.0f})"
    )

    # Area histogram (log scale for better visualization)
    axes[1, 0].hist(areas, bins=50, color=COLORS[2], edgecolor="white", alpha=0.8)
    axes[1, 0].set_xlabel("Area (pixels²)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title(
        f"Crop Area Distribution\n(mean: {np.mean(areas):,.0f}, median: {np.median(areas):,.0f})"
    )

    # Width vs Height scatter (sample if too many points)
    sample_size = min(10000, len(widths))
    indices = np.random.choice(len(widths), sample_size, replace=False)
    sample_w = [widths[i] for i in indices]
    sample_h = [heights[i] for i in indices]

    axes[1, 1].scatter(sample_w, sample_h, alpha=0.3, s=5, color=COLORS[3])
    axes[1, 1].set_xlabel("Width (pixels)")
    axes[1, 1].set_ylabel("Height (pixels)")
    axes[1, 1].set_title(f"Width vs Height (n={sample_size:,} sampled)")

    plt.tight_layout()
    fig.savefig(output_path / "04_crop_dimensions.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 04_crop_dimensions.png")


def plot_caption_statistics(output_path: Path):
    """Statistics about captions: length distribution, token count, language distribution."""
    caption_lengths = []
    word_counts = []
    token_counts = []
    languages = Counter()

    # Use cl100k_base encoding (GPT-4 family)
    encoding = tiktoken.get_encoding("cl100k_base")

    for cap in Caption.select(Caption.text, Caption.lang):
        if cap.text:
            caption_lengths.append(len(cap.text))
            word_counts.append(len(cap.text.split()))
            token_counts.append(len(encoding.encode(cap.text)))
        if cap.lang:
            languages[cap.lang] += 1

    if not caption_lengths:
        logger.warning("No caption data found")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Caption character length histogram
    axes[0, 0].hist(caption_lengths, bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
    axes[0, 0].set_xlabel("Character Count")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title(
        f"Caption Length Distribution\n(mean: {np.mean(caption_lengths):.0f}, median: {np.median(caption_lengths):.0f})"
    )

    # Word count histogram
    axes[0, 1].hist(word_counts, bins=50, color=COLORS[1], edgecolor="white", alpha=0.8)
    axes[0, 1].set_xlabel("Word Count")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title(
        f"Caption Word Count Distribution\n(mean: {np.mean(word_counts):.1f}, median: {np.median(word_counts):.0f})"
    )

    # Token count histogram
    axes[0, 2].hist(token_counts, bins=50, color=COLORS[2], edgecolor="white", alpha=0.8)
    axes[0, 2].set_xlabel("Token Count")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title(
        f"Caption Token Count Distribution\n(mean: {np.mean(token_counts):.1f}, median: {np.median(token_counts):.0f})"
    )

    # Language distribution
    if languages:
        lang_labels = [lang for lang, _ in languages.most_common(10)]
        lang_values = [count for _, count in languages.most_common(10)]

        bars = axes[1, 0].barh(lang_labels, lang_values, color=COLORS[: len(lang_labels)])
        axes[1, 0].set_xlabel("Number of Captions")
        axes[1, 0].set_title("Caption Language Distribution (Top 10)")

        for bar, val in zip(bars, lang_values):
            axes[1, 0].text(
                bar.get_width() + max(lang_values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}",
                va="center",
                fontsize=9,
            )
    else:
        axes[1, 0].text(
            0.5, 0.5, "No language data", ha="center", va="center", transform=axes[1, 0].transAxes
        )

    # Words vs Tokens scatter plot
    sample_size = min(5000, len(word_counts))
    indices = np.random.choice(len(word_counts), sample_size, replace=False)
    sample_words = [word_counts[i] for i in indices]
    sample_tokens = [token_counts[i] for i in indices]

    axes[1, 1].scatter(sample_words, sample_tokens, alpha=0.3, s=5, color=COLORS[3])
    axes[1, 1].set_xlabel("Word Count")
    axes[1, 1].set_ylabel("Token Count")
    axes[1, 1].set_title(f"Words vs Tokens (n={sample_size:,} sampled)")
    # Add diagonal reference line
    max_val = max(max(sample_words), max(sample_tokens))
    axes[1, 1].plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="1:1 ratio")
    axes[1, 1].legend()

    # Summary text
    axes[1, 2].axis("off")
    total_tokens = sum(token_counts)
    avg_tokens_per_word = np.mean([t / w for t, w in zip(token_counts, word_counts) if w > 0])

    summary_text = f"""Caption Statistics Summary

Total captions: {len(caption_lengths):,}
Unique languages: {len(languages)}

Character length:
  Mean: {np.mean(caption_lengths):.1f}
  Median: {np.median(caption_lengths):.0f}
  Min: {min(caption_lengths)}
  Max: {max(caption_lengths)}

Word count:
  Mean: {np.mean(word_counts):.1f}
  Median: {np.median(word_counts):.0f}
  Min: {min(word_counts)}
  Max: {max(word_counts)}

Token count (cl100k_base):
  Total: {total_tokens:,}
  Mean: {np.mean(token_counts):.1f}
  Median: {np.median(token_counts):.0f}
  Min: {min(token_counts)}
  Max: {max(token_counts)}
  Avg tokens/word: {avg_tokens_per_word:.2f}
"""
    axes[1, 2].text(
        0.1,
        0.95,
        summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path / "05_caption_statistics.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 05_caption_statistics.png")


def plot_deduplication_statistics(output_path: Path):
    """Statistics about deduplication: group sizes, duplicate rates."""
    # Hash deduplication groups
    hash_groups = defaultdict(int)
    for dh in DedupedHash.select(DedupedHash.group_id):
        hash_groups[dh.group_id] += 1

    # Embedding deduplication groups
    emb_groups = defaultdict(int)
    for de in DedupedEmbedding.select(DedupedEmbedding.group_id):
        emb_groups[de.group_id] += 1

    if not hash_groups and not emb_groups:
        logger.warning("No deduplication data found")
        return

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    # Hash group size distribution
    if hash_groups:
        hash_sizes = list(hash_groups.values())
        axes[0, 0].hist(
            hash_sizes, bins=min(50, max(hash_sizes)), color=COLORS[0], edgecolor="white", alpha=0.8
        )
        axes[0, 0].set_xlabel("Group Size")
        axes[0, 0].set_ylabel("Number of Groups")
        axes[0, 0].set_title(
            f"Hash Dedup Group Size Distribution\n({len(hash_groups):,} groups, {sum(hash_sizes):,} items)"
        )
        axes[0, 0].set_yscale("log")
    else:
        axes[0, 0].text(
            0.5, 0.5, "No hash dedup data", ha="center", va="center", transform=axes[0, 0].transAxes
        )

    # Embedding group size distribution
    if emb_groups:
        emb_sizes = list(emb_groups.values())
        axes[0, 1].hist(
            emb_sizes, bins=min(50, max(emb_sizes)), color=COLORS[1], edgecolor="white", alpha=0.8
        )
        axes[0, 1].set_xlabel("Group Size")
        axes[0, 1].set_ylabel("Number of Groups")
        axes[0, 1].set_title(
            f"Embedding Dedup Group Size Distribution\n({len(emb_groups):,} groups, {sum(emb_sizes):,} items)"
        )
        axes[0, 1].set_yscale("log")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No embedding dedup data",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )

    # Group size comparison (bar chart for small group sizes)
    if hash_groups:
        hash_size_counts = Counter(hash_groups.values())
        sizes = sorted([s for s in hash_size_counts.keys() if s <= 10])
        hash_counts = [hash_size_counts.get(s, 0) for s in sizes]

        x = np.arange(len(sizes))
        width = 0.35

        if emb_groups:
            emb_size_counts = Counter(emb_groups.values())
            emb_counts = [emb_size_counts.get(s, 0) for s in sizes]
            axes[1, 0].bar(x - width / 2, hash_counts, width, label="Hash", color=COLORS[0])
            axes[1, 0].bar(x + width / 2, emb_counts, width, label="Embedding", color=COLORS[1])
        else:
            axes[1, 0].bar(x, hash_counts, width, label="Hash", color=COLORS[0])

        axes[1, 0].set_xlabel("Group Size")
        axes[1, 0].set_ylabel("Number of Groups")
        axes[1, 0].set_title("Group Size Distribution (size 1-10)")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(sizes)
        axes[1, 0].legend()
        axes[1, 0].set_yscale("log")

    # Summary statistics
    axes[1, 1].axis("off")

    hash_sizes = list(hash_groups.values()) if hash_groups else []
    emb_sizes = list(emb_groups.values()) if emb_groups else []

    hash_singletons = sum(1 for s in hash_sizes if s == 1) if hash_sizes else 0
    emb_singletons = sum(1 for s in emb_sizes if s == 1) if emb_sizes else 0
    hash_duplicates = sum(1 for s in hash_sizes if s > 1) if hash_sizes else 0
    emb_duplicates = sum(1 for s in emb_sizes if s > 1) if emb_sizes else 0

    summary_text = f"""Deduplication Summary

Hash-based Deduplication:
  Total groups: {len(hash_groups):,}
  Total items: {sum(hash_sizes):,}
  Singleton groups: {hash_singletons:,}
  Duplicate groups: {hash_duplicates:,}
  Largest group: {max(hash_sizes) if hash_sizes else 0}
  Avg group size: {(np.mean(hash_sizes) if hash_sizes else 0):.2f}

Embedding-based Deduplication:
  Total groups: {len(emb_groups):,}
  Total items: {sum(emb_sizes):,}
  Singleton groups: {emb_singletons:,}
  Duplicate groups: {emb_duplicates:,}
  Largest group: {max(emb_sizes) if emb_sizes else 0}
  Avg group size: {(np.mean(emb_sizes) if emb_sizes else 0):.2f}
"""
    axes[1, 1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path / "06_deduplication_statistics.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 06_deduplication_statistics.png")


def plot_batch_progress(output_path: Path):
    """Visualization of batch progress: completed vs total."""
    all_batches = list(PipelineBatch.select())

    if not all_batches:
        logger.warning("No batch data found for progress visualization")
        return

    # Categorize batches
    completed_batches = [b for b in all_batches if b.started_date and b.ended_date]
    in_progress_batches = [b for b in all_batches if b.started_date and not b.ended_date]
    pending_batches = [b for b in all_batches if not b.started_date]
    crashed_batches = [b for b in all_batches if b.has_crashed]

    total = len(all_batches)
    completed = len(completed_batches)
    in_progress = len(in_progress_batches)
    pending = len(pending_batches)
    crashed = len(crashed_batches)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # Pie chart of batch status
    status_labels = []
    status_values = []
    status_colors = []

    if completed > 0:
        status_labels.append(f"Completed ({completed:,})")
        status_values.append(completed)
        status_colors.append(COLORS[2])  # Green-ish
    if in_progress > 0:
        status_labels.append(f"In Progress ({in_progress:,})")
        status_values.append(in_progress)
        status_colors.append(COLORS[1])  # Yellow-ish
    if pending > 0:
        status_labels.append(f"Pending ({pending:,})")
        status_values.append(pending)
        status_colors.append(COLORS[0])  # Light color
    if crashed > 0:
        status_labels.append(f"Crashed ({crashed:,})")
        status_values.append(crashed)
        status_colors.append(COLORS[3])  # Red-ish

    if status_values:
        axes[0].pie(
            status_values,
            labels=status_labels,
            autopct="%1.1f%%",
            colors=status_colors,
            startangle=90,
        )
        axes[0].set_title(f"Batch Status Overview\n({total:,} total batches)")

    # Progress bar style visualization
    axes[1].barh(
        ["Progress"],
        [completed],
        color=COLORS[2],
        label=f"Completed: {completed:,}",
        height=0.4,
    )
    axes[1].barh(
        ["Progress"],
        [in_progress],
        left=[completed],
        color=COLORS[1],
        label=f"In Progress: {in_progress:,}",
        height=0.4,
    )
    axes[1].barh(
        ["Progress"],
        [pending],
        left=[completed + in_progress],
        color=COLORS[0],
        label=f"Pending: {pending:,}",
        height=0.4,
    )

    axes[1].set_xlim(0, total)
    axes[1].set_xlabel("Number of Batches")
    axes[1].set_title(
        f"Batch Progress: {completed:,}/{total:,} ({100*completed/total:.1f}% complete)"
    )
    axes[1].legend(loc="upper right")

    # Add percentage text in the middle of the bar
    if total > 0:
        axes[1].text(
            total / 2,
            0,
            f"{100*completed/total:.1f}%",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="black",
        )

    plt.tight_layout()
    fig.savefig(output_path / "07_batch_progress.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 07_batch_progress.png")


def plot_pipeline_statistics(output_path: Path):
    """Statistics about pipeline runs and completed batches."""
    runs = list(PipelineRun.select())
    all_batches = list(PipelineBatch.select())

    # Filter to only completed batches (have both start and end time)
    completed_batches = [b for b in all_batches if b.started_date and b.ended_date]

    if not runs and not all_batches:
        logger.warning("No pipeline data found")
        return

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    # Batch processing times (only from completed batches)
    processing_times = []
    for batch in completed_batches:
        duration = (batch.ended_date - batch.started_date).total_seconds() / 60  # minutes
        if duration > 0:
            processing_times.append(duration)

    if processing_times:
        axes[0, 0].hist(processing_times, bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
        axes[0, 0].set_xlabel("Processing Time (minutes)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title(
            f"Batch Processing Time Distribution\n(n={len(completed_batches):,} completed, mean: {np.mean(processing_times):.1f} min)"
        )
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "No completed batch data",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_title("Batch Processing Time Distribution")

    # Crash rate (only among completed batches)
    crash_counts = sum(1 for b in completed_batches if b.has_crashed)
    success_counts = len(completed_batches) - crash_counts

    if completed_batches:
        axes[0, 1].pie(
            [success_counts, crash_counts],
            labels=["Success", "Crashed"],
            autopct="%1.1f%%",
            colors=[COLORS[2], COLORS[3]],
            explode=(0, 0.1) if crash_counts > 0 else (0, 0),
        )
        axes[0, 1].set_title(
            f"Completed Batch Success Rate\n({len(completed_batches):,} completed batches)"
        )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No completed batch data",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )

    # Batches per node (only completed batches)
    node_counts = Counter(b.node_name for b in completed_batches if b.node_name)
    if node_counts:
        nodes = [n for n, _ in node_counts.most_common()]
        counts = [c for _, c in node_counts.most_common()]
        axes[1, 0].barh(nodes, counts, color=COLORS[: len(nodes)])
        axes[1, 0].set_xlabel("Number of Completed Batches")
        axes[1, 0].set_title("Completed Batches by Node")
    else:
        axes[1, 0].text(
            0.5, 0.5, "No node data", ha="center", va="center", transform=axes[1, 0].transAxes
        )

    # Summary statistics
    axes[1, 1].axis("off")

    # Pre-compute values for summary
    total_batches = len(all_batches)
    completed_count = len(completed_batches)
    completion_pct = f"{100*completed_count/total_batches:.1f}%" if total_batches else "N/A"
    success_pct = f"{100*success_counts/completed_count:.1f}%" if completed_count else "N/A"
    crash_pct = f"{100*crash_counts/completed_count:.1f}%" if completed_count else "N/A"
    mean_time = f"{np.mean(processing_times):.1f}" if processing_times else "N/A"
    median_time = f"{np.median(processing_times):.1f}" if processing_times else "N/A"
    min_time = f"{min(processing_times):.1f}" if processing_times else "N/A"
    max_time = f"{max(processing_times):.1f}" if processing_times else "N/A"

    summary_text = f"""Pipeline Statistics Summary
(Completed Batches Only)

Pipeline Runs: {len(runs):,}
Total Batches: {total_batches:,}
Completed Batches: {completed_count:,} ({completion_pct})
Unique Nodes: {len(node_counts)}

Completed Batch Status:
  Successful: {success_counts:,} ({success_pct})
  Crashed: {crash_counts:,} ({crash_pct})

Processing Time (minutes):
  Mean: {mean_time}
  Median: {median_time}
  Min: {min_time}
  Max: {max_time}
"""
    axes[1, 1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path / "08_pipeline_statistics.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 08_pipeline_statistics.png")


def plot_volume_metadata(output_path: Path):
    """Statistics about volume metadata: topics, dates, languages."""
    topics = Counter()
    dates = Counter()
    languages = Counter()

    for vol in IBVolume.select(IBVolume.metadata):
        metadata = vol.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        elif metadata is None:
            metadata = {}

        topic = metadata.get("topic_or_subject_gen")
        if topic:
            topics[topic] += 1

        date = metadata.get("date1_src")
        if date:
            # Try to extract year
            try:
                year = int(str(date)[:4])
                if 1400 <= year <= 2100:  # Reasonable year range
                    dates[year] += 1
            except (ValueError, TypeError):
                pass

        lang = metadata.get("language_src")
        if lang:
            languages[lang] += 1

    if not topics and not dates and not languages:
        logger.warning("No volume metadata found")
        return

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LARGE)

    # Topic distribution (top 15)
    if topics:
        topic_labels = [t[:30] + "..." if len(t) > 30 else t for t, _ in topics.most_common(15)]
        topic_values = [c for _, c in topics.most_common(15)]

        bars = axes[0, 0].barh(topic_labels, topic_values, color=COLORS[0])
        axes[0, 0].set_xlabel("Number of Volumes")
        axes[0, 0].set_title(f"Topic Distribution (Top 15 of {len(topics)})")
        axes[0, 0].tick_params(axis="y", labelsize=8)
    else:
        axes[0, 0].text(
            0.5, 0.5, "No topic data", ha="center", va="center", transform=axes[0, 0].transAxes
        )

    # Publication date distribution
    if dates:
        years = sorted(dates.keys())
        counts = [dates[y] for y in years]

        axes[0, 1].fill_between(years, counts, alpha=0.7, color=COLORS[1])
        axes[0, 1].plot(years, counts, color=COLORS[1], linewidth=2)
        axes[0, 1].set_xlabel("Publication Year")
        axes[0, 1].set_ylabel("Number of Volumes")
        axes[0, 1].set_title(
            f"Publication Date Distribution\n(n={sum(counts):,} volumes with date)"
        )
    else:
        axes[0, 1].text(
            0.5, 0.5, "No date data", ha="center", va="center", transform=axes[0, 1].transAxes
        )

    # Language distribution
    if languages:
        lang_labels = [l for l, _ in languages.most_common(15)]
        lang_values = [c for _, c in languages.most_common(15)]

        bars = axes[1, 0].barh(lang_labels, lang_values, color=COLORS[2])
        axes[1, 0].set_xlabel("Number of Volumes")
        axes[1, 0].set_title(f"Language Distribution (Top 15 of {len(languages)})")
    else:
        axes[1, 0].text(
            0.5, 0.5, "No language data", ha="center", va="center", transform=axes[1, 0].transAxes
        )

    # Summary
    axes[1, 1].axis("off")
    total_volumes = IBVolume.select().count()

    summary_text = f"""Volume Metadata Summary

Total volumes: {total_volumes:,}

Topics:
  Unique topics: {len(topics)}
  Most common: {topics.most_common(1)[0][0][:40] if topics else 'N/A'}

Publication Dates:
  Volumes with date: {sum(dates.values()) if dates else 0:,}
  Year range: {min(dates.keys()) if dates else 'N/A'} - {max(dates.keys()) if dates else 'N/A'}
  Median year: {int(np.median(list(dates.keys()))) if dates else 'N/A'}

Languages:
  Unique languages: {len(languages)}
  Most common: {languages.most_common(1)[0][0] if languages else 'N/A'}
"""
    axes[1, 1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path / "09_volume_metadata.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 09_volume_metadata.png")


def plot_detections_per_volume(output_path: Path):
    """Distribution of detections per volume."""
    # Count detections per volume
    volume_detection_counts = Counter()

    for item in PipelineBatchItem.select(
        PipelineBatchItem.id_pipeline_batch_item, PipelineBatchItem.ib_volume
    ):
        det_count = (
            Detection.select()
            .where(Detection.pipeline_batch_item == item.id_pipeline_batch_item)
            .count()
        )
        volume_detection_counts[item.ib_volume.barcode] = det_count

    if not volume_detection_counts:
        logger.warning("No detection data found")
        return

    counts = list(volume_detection_counts.values())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_LARGE)

    # Histogram of detections per volume
    axes[0].hist(counts, bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Number of Detections")
    axes[0].set_ylabel("Number of Volumes")
    axes[0].set_title(f"Detections per Volume\n(n={len(counts):,} volumes)")
    axes[0].axvline(
        np.mean(counts), color="red", linestyle="--", label=f"Mean: {np.mean(counts):.1f}"
    )
    axes[0].axvline(
        np.median(counts), color="orange", linestyle="--", label=f"Median: {np.median(counts):.0f}"
    )
    axes[0].legend()

    # Summary statistics
    axes[1].axis("off")

    zero_detection_vols = sum(1 for c in counts if c == 0)
    high_detection_vols = sum(1 for c in counts if c > 100)

    summary_text = f"""Detections per Volume Summary

Total volumes processed: {len(counts):,}
Total detections: {sum(counts):,}

Detections per volume:
  Mean: {np.mean(counts):.1f}
  Median: {np.median(counts):.0f}
  Std Dev: {np.std(counts):.1f}
  Min: {min(counts)}
  Max: {max(counts)}

Distribution:
  Volumes with 0 detections: {zero_detection_vols:,}
  Volumes with >100 detections: {high_detection_vols:,}

Percentiles:
  25th: {np.percentile(counts, 25):.0f}
  50th: {np.percentile(counts, 50):.0f}
  75th: {np.percentile(counts, 75):.0f}
  90th: {np.percentile(counts, 90):.0f}
  99th: {np.percentile(counts, 99):.0f}
"""
    axes[1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path / "10_detections_per_volume.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved: 10_detections_per_volume.png")


def write_summary_report(counts: dict, output_path: Path):
    """Write a text summary report."""
    report_file = output_path / "summary_report.txt"

    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATABASE STATISTICS SUMMARY REPORT\n")
        f.write(f"Generated: {DATETIME_SLUG}\n")
        f.write("=" * 80 + "\n\n")

        f.write("TABLE RECORD COUNTS\n")
        f.write("-" * 40 + "\n")
        for table, count in counts.items():
            f.write(f"  {table}: {count:,}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("GENERATED VISUALIZATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("  01_table_counts.png - Database table record counts\n")
        f.write("  02_classification_distribution.png - Classification class distribution\n")
        f.write("  03_confidence_distributions.png - Detection and classification confidence\n")
        f.write("  04_crop_dimensions.png - Crop width, height, and area distributions\n")
        f.write("  05_caption_statistics.png - Caption length, token count, and language statistics\n")
        f.write("  06_deduplication_statistics.png - Deduplication group statistics\n")
        f.write("  07_batch_progress.png - Batch progress (completed vs total)\n")
        f.write("  08_pipeline_statistics.png - Pipeline run and completed batch statistics\n")
        f.write("  09_volume_metadata.png - Volume metadata distributions\n")
        f.write("  10_detections_per_volume.png - Detections per volume distribution\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("DATA EXPORT\n")
        f.write("-" * 40 + "\n")
        f.write("  statistics.json - All computed statistics in JSON format\n")

    logger.info(f"Summary report written to: {report_file}")


def collect_all_statistics(skip_slow: bool = False) -> dict:
    """Collect all statistics into a dictionary for JSON export."""
    stats_data = {
        "generated_at": DATETIME_SLUG,
        "table_counts": get_table_counts(),
    }

    # Classification distribution
    class_counts = Counter()
    for cls in Classification.select(Classification.pred_class):
        class_counts[cls.pred_class] += 1
    stats_data["classification_distribution"] = {
        CLASSIFICATION_CLASS_DICT.get(str(k), f"unknown_{k}"): v
        for k, v in class_counts.items()
    }

    # Confidence scores
    det_confs = [
        d.bbox_conf for d in Detection.select(Detection.bbox_conf) if d.bbox_conf is not None
    ]
    cls_confs = [
        c.pred_conf for c in Classification.select(Classification.pred_conf) if c.pred_conf is not None
    ]
    stats_data["confidence_scores"] = {
        "detection": {
            "count": len(det_confs),
            "mean": float(np.mean(det_confs)) if det_confs else None,
            "median": float(np.median(det_confs)) if det_confs else None,
            "min": float(min(det_confs)) if det_confs else None,
            "max": float(max(det_confs)) if det_confs else None,
            "std": float(np.std(det_confs)) if det_confs else None,
        },
        "classification": {
            "count": len(cls_confs),
            "mean": float(np.mean(cls_confs)) if cls_confs else None,
            "median": float(np.median(cls_confs)) if cls_confs else None,
            "min": float(min(cls_confs)) if cls_confs else None,
            "max": float(max(cls_confs)) if cls_confs else None,
            "std": float(np.std(cls_confs)) if cls_confs else None,
        },
    }

    # Crop dimensions
    widths, heights, areas = [], [], []
    for det in Detection.select(Detection.bbox_xywh):
        if det.bbox_xywh and len(det.bbox_xywh) == 4:
            w, h = det.bbox_xywh[2], det.bbox_xywh[3]
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
    stats_data["crop_dimensions"] = {
        "count": len(widths),
        "width": {
            "mean": float(np.mean(widths)) if widths else None,
            "median": float(np.median(widths)) if widths else None,
            "min": float(min(widths)) if widths else None,
            "max": float(max(widths)) if widths else None,
            "std": float(np.std(widths)) if widths else None,
        },
        "height": {
            "mean": float(np.mean(heights)) if heights else None,
            "median": float(np.median(heights)) if heights else None,
            "min": float(min(heights)) if heights else None,
            "max": float(max(heights)) if heights else None,
            "std": float(np.std(heights)) if heights else None,
        },
        "area": {
            "mean": float(np.mean(areas)) if areas else None,
            "median": float(np.median(areas)) if areas else None,
            "min": float(min(areas)) if areas else None,
            "max": float(max(areas)) if areas else None,
            "std": float(np.std(areas)) if areas else None,
        },
    }

    # Caption statistics
    caption_lengths, word_counts, token_counts = [], [], []
    languages = Counter()
    encoding = tiktoken.get_encoding("cl100k_base")
    for cap in Caption.select(Caption.text, Caption.lang):
        if cap.text:
            caption_lengths.append(len(cap.text))
            word_counts.append(len(cap.text.split()))
            token_counts.append(len(encoding.encode(cap.text)))
        if cap.lang:
            languages[cap.lang] += 1
    stats_data["captions"] = {
        "count": len(caption_lengths),
        "character_length": {
            "mean": float(np.mean(caption_lengths)) if caption_lengths else None,
            "median": float(np.median(caption_lengths)) if caption_lengths else None,
            "min": int(min(caption_lengths)) if caption_lengths else None,
            "max": int(max(caption_lengths)) if caption_lengths else None,
            "std": float(np.std(caption_lengths)) if caption_lengths else None,
        },
        "word_count": {
            "mean": float(np.mean(word_counts)) if word_counts else None,
            "median": float(np.median(word_counts)) if word_counts else None,
            "min": int(min(word_counts)) if word_counts else None,
            "max": int(max(word_counts)) if word_counts else None,
            "std": float(np.std(word_counts)) if word_counts else None,
        },
        "token_count": {
            "total": int(sum(token_counts)) if token_counts else None,
            "mean": float(np.mean(token_counts)) if token_counts else None,
            "median": float(np.median(token_counts)) if token_counts else None,
            "min": int(min(token_counts)) if token_counts else None,
            "max": int(max(token_counts)) if token_counts else None,
            "std": float(np.std(token_counts)) if token_counts else None,
            "encoding": "cl100k_base",
        },
        "language_distribution": dict(languages.most_common()),
    }

    # Deduplication statistics
    hash_groups = defaultdict(int)
    for dh in DedupedHash.select(DedupedHash.group_id):
        hash_groups[dh.group_id] += 1
    emb_groups = defaultdict(int)
    for de in DedupedEmbedding.select(DedupedEmbedding.group_id):
        emb_groups[de.group_id] += 1

    hash_sizes = list(hash_groups.values())
    emb_sizes = list(emb_groups.values())
    stats_data["deduplication"] = {
        "hash_based": {
            "total_groups": len(hash_groups),
            "total_items": sum(hash_sizes) if hash_sizes else 0,
            "singleton_groups": sum(1 for s in hash_sizes if s == 1),
            "duplicate_groups": sum(1 for s in hash_sizes if s > 1),
            "largest_group": max(hash_sizes) if hash_sizes else 0,
            "avg_group_size": float(np.mean(hash_sizes)) if hash_sizes else None,
        },
        "embedding_based": {
            "total_groups": len(emb_groups),
            "total_items": sum(emb_sizes) if emb_sizes else 0,
            "singleton_groups": sum(1 for s in emb_sizes if s == 1),
            "duplicate_groups": sum(1 for s in emb_sizes if s > 1),
            "largest_group": max(emb_sizes) if emb_sizes else 0,
            "avg_group_size": float(np.mean(emb_sizes)) if emb_sizes else None,
        },
    }

    # Pipeline/batch statistics
    all_batches = list(PipelineBatch.select())
    completed_batches = [b for b in all_batches if b.started_date and b.ended_date]
    processing_times = []
    for batch in completed_batches:
        duration = (batch.ended_date - batch.started_date).total_seconds() / 60
        if duration > 0:
            processing_times.append(duration)

    crash_counts = sum(1 for b in completed_batches if b.has_crashed)
    node_counts = Counter(b.node_name for b in completed_batches if b.node_name)

    stats_data["pipeline"] = {
        "total_runs": PipelineRun.select().count(),
        "batches": {
            "total": len(all_batches),
            "completed": len(completed_batches),
            "in_progress": len([b for b in all_batches if b.started_date and not b.ended_date]),
            "pending": len([b for b in all_batches if not b.started_date]),
            "crashed": crash_counts,
            "success_rate": float(len(completed_batches) - crash_counts) / len(completed_batches) if completed_batches else None,
        },
        "processing_time_minutes": {
            "mean": float(np.mean(processing_times)) if processing_times else None,
            "median": float(np.median(processing_times)) if processing_times else None,
            "min": float(min(processing_times)) if processing_times else None,
            "max": float(max(processing_times)) if processing_times else None,
            "std": float(np.std(processing_times)) if processing_times else None,
        },
        "batches_by_node": dict(node_counts),
    }

    # Volume metadata
    topics = Counter()
    dates = Counter()
    vol_languages = Counter()
    for vol in IBVolume.select(IBVolume.metadata):
        metadata = vol.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        elif metadata is None:
            metadata = {}

        topic = metadata.get("topic_or_subject_gen")
        if topic:
            topics[topic] += 1
        date = metadata.get("date1_src")
        if date:
            try:
                year = int(str(date)[:4])
                if 1400 <= year <= 2100:
                    dates[year] += 1
            except (ValueError, TypeError):
                pass
        lang = metadata.get("language_src")
        if lang:
            vol_languages[lang] += 1

    stats_data["volume_metadata"] = {
        "total_volumes": IBVolume.select().count(),
        "topics": {
            "unique_count": len(topics),
            "distribution": dict(topics.most_common(50)),
        },
        "publication_dates": {
            "volumes_with_date": sum(dates.values()),
            "year_range": [min(dates.keys()), max(dates.keys())] if dates else None,
            "median_year": int(np.median(list(dates.keys()))) if dates else None,
            "distribution": dict(sorted(dates.items())),
        },
        "languages": {
            "unique_count": len(vol_languages),
            "distribution": dict(vol_languages.most_common()),
        },
    }

    # Detections per volume (slow query)
    if not skip_slow:
        volume_detection_counts = Counter()
        for item in PipelineBatchItem.select(
            PipelineBatchItem.id_pipeline_batch_item, PipelineBatchItem.ib_volume
        ):
            det_count = (
                Detection.select()
                .where(Detection.pipeline_batch_item == item.id_pipeline_batch_item)
                .count()
            )
            volume_detection_counts[item.ib_volume.barcode] = det_count

        counts = list(volume_detection_counts.values())
        stats_data["detections_per_volume"] = {
            "volumes_processed": len(counts),
            "total_detections": sum(counts) if counts else 0,
            "mean": float(np.mean(counts)) if counts else None,
            "median": float(np.median(counts)) if counts else None,
            "min": int(min(counts)) if counts else None,
            "max": int(max(counts)) if counts else None,
            "std": float(np.std(counts)) if counts else None,
            "volumes_with_zero": sum(1 for c in counts if c == 0),
            "volumes_with_over_100": sum(1 for c in counts if c > 100),
            "percentiles": {
                "25th": float(np.percentile(counts, 25)) if counts else None,
                "50th": float(np.percentile(counts, 50)) if counts else None,
                "75th": float(np.percentile(counts, 75)) if counts else None,
                "90th": float(np.percentile(counts, 90)) if counts else None,
                "99th": float(np.percentile(counts, 99)) if counts else None,
            },
        }

    return stats_data


def save_statistics_json(stats_data: dict, output_path: Path):
    """Save all statistics to a JSON file."""
    json_file = output_path / "statistics.json"
    with open(json_file, "w") as f:
        json.dump(stats_data, f, indent=2)
    logger.info(f"Statistics JSON saved to: {json_file}")


@click.command("stats")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=ANALYSIS_OUTPUT_DIR,
    help="Output directory for statistics visualizations",
)
@click.option(
    "--skip-slow/--no-skip-slow",
    default=False,
    help="Skip slow queries (detections per volume)",
)
def stats(output_dir, skip_slow):
    """
    Generate aggregate statistics and visualizations from the database.

    Creates PNG charts and a JSON summary covering:
    - Table record counts
    - Classification distributions
    - Confidence scores
    - Crop dimensions
    - Caption statistics
    - Deduplication effectiveness
    - Pipeline performance
    - Volume metadata
    - Detections per volume

    Examples:
        stats
        stats --output-dir ./my_stats
        stats --skip-slow
    """
    logger.info("Starting database statistics generation...")

    # Get database connection
    db = get_db()

    # Create output directory
    output_path = Path(output_dir) / f"stats_{DATETIME_SLUG}"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_path}")

    # Get table counts first (used for overview)
    logger.info("Fetching table counts...")
    counts = get_table_counts()

    for table, count in counts.items():
        logger.info(f"  {table}: {count:,}")

    # Generate visualizations
    logger.info("Generating visualizations...")

    plot_table_counts(counts, output_path)
    plot_classification_distribution(output_path)
    plot_confidence_distributions(output_path)
    plot_crop_dimensions(output_path)
    plot_caption_statistics(output_path)
    plot_deduplication_statistics(output_path)
    plot_batch_progress(output_path)
    plot_pipeline_statistics(output_path)
    plot_volume_metadata(output_path)

    if not skip_slow:
        logger.info("Generating detections per volume (this may take a while)...")
        plot_detections_per_volume(output_path)
    else:
        logger.info("Skipping detections per volume (--skip-slow)")

    # Write summary report
    write_summary_report(counts, output_path)

    # Collect and save all statistics to JSON
    logger.info("Collecting statistics for JSON export...")
    stats_data = collect_all_statistics(skip_slow=skip_slow)
    save_statistics_json(stats_data, output_path)

    logger.success(f"Statistics generation complete! Results saved to: {output_path}")
