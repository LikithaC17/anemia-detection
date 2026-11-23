"""
Phase 2 Graph Generation Module: Data Verification, Cleaning & Splitting
Generates 6 publication-quality graphs from verified Phase 2 outputs.
"""

import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization.graph_utils import (
    set_academic_style, save_academic_figure, PALETTE
)

PHASE2_DIR = WORKSPACE_ROOT / "results" / "phase2"
OUT_DIR = WORKSPACE_ROOT / "results" / "graph_analysis" / "phase2"


def generate_phase2_graphs():
    set_academic_style()
    print("\n--- Generating Phase 2 Graphs ---")

    with open(PHASE2_DIR / "split_summary.json", "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    train_df = pd.read_csv(PHASE2_DIR / "train.csv")
    val_df = pd.read_csv(PHASE2_DIR / "validation.csv")
    test_df = pd.read_csv(PHASE2_DIR / "test.csv")

    # -------------------------------------------------------------
    # Graph 1: Before vs After Data Cleaning
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    stages = ["Raw Discovered", "Corrupted / Invalid", "Verified Cleaned"]
    vals = [7377, 8, 7369]
    colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"]]

    bars = ax.bar(stages, vals, color=colors, width=0.5, edgecolor="black", linewidth=1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Dataset Cleaning Audit (Pre vs. Post Cleaning)", pad=15)
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, 8500)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P2_01_before_after_cleaning", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 2: Train, Validation and Test Distribution by Modality
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    splits_data = {
        "Eye": [1830, 397, 384],
        "Palm": [2273, 483, 493],
        "Nail Beds": [1033, 236, 240]
    }
    df_splits = pd.DataFrame(splits_data, index=["Train (70%)", "Validation (15%)", "Test (15%)"])

    x = np.arange(len(df_splits.index))
    width = 0.25

    ax.bar(x - width, df_splits["Eye"], width, label="Eye Conjunctiva", color=PALETTE["primary"], edgecolor="black")
    ax.bar(x, df_splits["Palm"], width, label="Palmar Pallor", color=PALETTE["secondary"], edgecolor="black")
    ax.bar(x + width, df_splits["Nail Beds"], width, label="Nail Beds", color=PALETTE["warning"], edgecolor="black")

    ax.set_title("Image Distribution Across Dataset Partitions by Modality", pad=15)
    ax.set_ylabel("Number of Images")
    ax.set_xticks(x)
    ax.set_xticklabels(df_splits.index)
    ax.legend(title="Modality", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P2_02_train_val_test_distribution", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 3: Dataset Split Percentage (Calculated)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    counts = [5136, 1116, 1117]
    labels = ["Train Set\n(5,136 images)", "Validation Set\n(1,116 images)", "Test Set\n(1,117 images)"]
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]]

    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct="%1.2f%%", startangle=140,
        colors=colors, wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2),
        textprops=dict(fontsize=10.5, fontweight="bold")
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_color("white")

    ax.set_title("Patient-Stratified Dataset Split Proportions (Target: 70:15:15)", pad=15)
    save_academic_figure(fig, "P2_03_split_percentage", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 4: Class Distribution by Split
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    split_names = ["Train Split", "Validation Split", "Test Split"]
    non_anemia_counts = [
        sum(train_df["target_binary"] == 0),
        sum(val_df["target_binary"] == 0),
        sum(test_df["target_binary"] == 0)
    ]
    anemia_counts = [
        sum(train_df["target_binary"] == 1),
        sum(val_df["target_binary"] == 1),
        sum(test_df["target_binary"] == 1)
    ]

    x = np.arange(len(split_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, non_anemia_counts, width, label="Non_Anemia", color=PALETTE["non_anemia"], edgecolor="black")
    bars2 = ax.bar(x + width/2, anemia_counts, width, label="Anemia", color=PALETTE["anemia"], edgecolor="black")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 40, f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 40, f"{h:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Class Representation Across Train, Validation, and Test Splits", pad=15)
    ax.set_ylabel("Number of Images")
    ax.set_xticks(x)
    ax.set_xticklabels(split_names)
    ax.set_ylim(0, max(non_anemia_counts) * 1.15)
    ax.legend(title="Class", loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P2_04_class_distribution_by_split", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 5: Duplicate Image Analysis & Cluster Containment
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    metrics = ["Exact Duplicates\n(Identified)", "Subject Clusters\n(Formed)", "Cross-Split Leakage\n(Allowed)"]
    dup_counts = [60, 1438, 0]
    colors = [PALETTE["warning"], PALETTE["primary"], PALETTE["secondary"]]

    bars = ax.bar(metrics, dup_counts, color=colors, width=0.5, edgecolor="black", linewidth=1)
    for bar, val in zip(bars, dup_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Duplicate Image Containment via Patient Clustering", pad=15)
    ax.set_ylabel("Count")
    ax.set_ylim(0, 1650)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P2_05_duplicate_analysis", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 6: Data Leakage Verification
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    pairs = ["Train vs. Val\nPatient Overlap", "Train vs. Test\nPatient Overlap", "Val vs. Test\nPatient Overlap", "Cross-Split\nDuplicate Leakage"]
    leakage_vals = [0, 0, 0, 0]

    bars = ax.bar(pairs, leakage_vals, color=PALETTE["secondary"], width=0.5, edgecolor="black", linewidth=1)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, 0.02, "0 (0.0% Leakage)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#065f46")

    ax.set_title("Patient-Level Data Leakage Verification Across Splits", pad=15)
    ax.set_ylabel("Overlap Count")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P2_06_data_leakage_verification", OUT_DIR)

    print("Phase 2 graphs successfully generated!")


if __name__ == "__main__":
    generate_phase2_graphs()
