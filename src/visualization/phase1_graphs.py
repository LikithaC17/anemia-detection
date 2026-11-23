"""
Phase 1 Graph Generation Module: Dataset Inspection & Metadata Analysis
Generates 5 publication-quality graphs from verified Phase 1 outputs.
"""

import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.graph_utils import (
    set_academic_style, save_academic_figure, PALETTE
)

PHASE1_DIR = WORKSPACE_ROOT / "results" / "phase1"
OUT_DIR = WORKSPACE_ROOT / "results" / "graph_analysis" / "phase1"


def generate_phase1_graphs():
    set_academic_style()
    print("\n--- Generating Phase 1 Graphs ---")

    meta_df = pd.read_csv(PHASE1_DIR / "dataset_metadata.csv")
    class_df = pd.read_csv(PHASE1_DIR / "class_distribution.csv")
    summary_df = pd.read_csv(PHASE1_DIR / "dataset_summary.csv")

    # -------------------------------------------------------------
    # Graph 1: Modality-wise Image Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    mod_counts = meta_df["modality"].value_counts()
    modalities = mod_counts.index.tolist()
    counts = mod_counts.values.tolist()
    total = sum(counts)

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]]
    bars = ax.bar(modalities, counts, color=colors, width=0.55, edgecolor="black", linewidth=1)

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Modality-wise Image Distribution (Total N = 7,377)", pad=15)
    ax.set_xlabel("Anatomical Modality")
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P1_01_modality_distribution", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 2: Class Distribution (Overall & Modality-wise)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cross_tab = pd.crosstab(meta_df["modality"], meta_df["class_label"])
    
    x = np.arange(len(cross_tab.index))
    width = 0.35

    bars1 = ax.bar(x - width/2, cross_tab["Non_Anemia"], width, label="Non_Anemia",
                   color=PALETTE["non_anemia"], edgecolor="black", linewidth=1)
    bars2 = ax.bar(x + width/2, cross_tab["Anemia"], width, label="Anemia",
                   color=PALETTE["anemia"], edgecolor="black", linewidth=1)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 30, f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 30, f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Class Distribution Across Anatomical Modalities", pad=15)
    ax.set_xlabel("Anatomical Modality")
    ax.set_ylabel("Number of Images")
    ax.set_xticks(x)
    ax.set_xticklabels(cross_tab.index)
    ax.set_ylim(0, max(cross_tab.max()) * 1.18)
    ax.legend(title="Class Label", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P1_02_class_distribution", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 3: Image Resolution Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(data=meta_df, x="width", y="height", hue="modality", alpha=0.6,
                    palette=[PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]],
                    s=35, ax=ax, edgecolor="none")
    ax.set_title("Image Resolution Distribution by Modality (Width vs. Height)", pad=15)
    ax.set_xlabel("Image Width (Pixels)")
    ax.set_ylabel("Image Height (Pixels)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Modality", loc="upper right")
    save_academic_figure(fig, "P1_03_resolution_distribution", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 4: Image Quality & File Size Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    meta_df["size_kb"] = meta_df["file_size_bytes"] / 1024.0
    sns.boxplot(data=meta_df, x="modality", y="size_kb", hue="format",
                palette=[PALETTE["teal"], PALETTE["warning"]], ax=ax, width=0.5, fliersize=2)
    ax.set_title("Image Quality & File Size Distribution (KB) by Format", pad=15)
    ax.set_xlabel("Anatomical Modality")
    ax.set_ylabel("File Size (KB)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Image Format", loc="upper right")
    save_academic_figure(fig, "P1_04_quality_distribution", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 5: Dataset Extraction Summary
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    categories = ["Discovered\nImages", "Verified\nUsable", "Exact\nDuplicates", "Corrupted\n/ Excluded"]
    counts_ext = [7377, 7369, 60, 8]
    bar_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"], PALETTE["accent"]]

    bars = ax.bar(categories, counts_ext, color=bar_colors, width=0.55, edgecolor="black", linewidth=1)
    for bar, val in zip(bars, counts_ext):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Phase 1 Dataset Extraction & Verification Summary", pad=15)
    ax.set_ylabel("Count")
    ax.set_ylim(0, 8500)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P1_05_extraction_summary", OUT_DIR)

    print("Phase 1 graphs successfully generated!")


if __name__ == "__main__":
    generate_phase1_graphs()
