"""
Phase 4 Graph Generation Module: Deep Learning Validation Results & Dynamics
Generates 10 publication-quality graphs strictly using Phase 4 training histories and validation metrics.
No Phase 5 test results are used.
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
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.visualization.graph_utils import (
    set_academic_style, save_academic_figure, PALETTE
)
from src.deep_learning.data_loader import create_dataloaders
from src.deep_learning.custom_cnn import get_custom_cnn
from src.deep_learning.resnet18_model import get_resnet18_transfer_model
from src.deep_learning.generate_phase4_final_artifacts import evaluate_checkpoint

PHASE4_DIR = WORKSPACE_ROOT / "results" / "phase4"
OUT_DIR = WORKSPACE_ROOT / "results" / "graph_analysis" / "phase4"
PHASE2_DIR = WORKSPACE_ROOT / "results" / "phase2"


def generate_phase4_graphs():
    set_academic_style()
    print("\n--- Generating Phase 4 Graphs ---")

    # Load Deep Learning comparison metrics
    dl_comp_df = pd.read_csv(PHASE4_DIR / "reports" / "deep_learning_comparison.csv")

    # Load Training Histories
    with open(PHASE4_DIR / "training_history" / "history_custom_cnn_eye.json") as f:
        cnn_hist = json.load(f)
    with open(PHASE4_DIR / "training_history" / "history_resnet18_eye.json") as f:
        res_hist = json.load(f)

    epochs = np.arange(1, len(cnn_hist["train_loss"]) + 1)

    # -------------------------------------------------------------
    # Graph 1: Custom CNN Training vs Validation Accuracy Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, [a * 100 for a in cnn_hist["train_acc"]], "o-", color=PALETTE["primary"],
            linewidth=2.2, label="Training Accuracy")
    ax.plot(epochs, [a * 100 for a in cnn_hist["val_acc"]], "s--", color=PALETTE["secondary"],
            linewidth=2.2, label="Validation Accuracy")
    ax.axvline(6, color="gray", linestyle=":", label="Best Epoch (Epoch 6)")
    ax.set_title("Custom CNN: Training vs. Validation Accuracy (Epochs 1–6)", pad=15)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(20, 100)
    ax.set_xticks(epochs)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_01_custom_cnn_accuracy", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 2: Custom CNN Training vs Validation Loss Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, cnn_hist["train_loss"], "o-", color=PALETTE["accent"],
            linewidth=2.2, label="Training Cross-Entropy Loss")
    ax.plot(epochs, cnn_hist["val_loss"], "s--", color=PALETTE["warning"],
            linewidth=2.2, label="Validation Loss")
    ax.axvline(6, color="gray", linestyle=":", label="Best Checkpoint (Epoch 6)")
    ax.set_title("Custom CNN: Training vs. Validation Loss Dynamics", pad=15)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_02_custom_cnn_loss", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 3: ResNet-18 Training vs Validation Accuracy Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, [a * 100 for a in res_hist["train_acc"]], "o-", color=PALETTE["primary"],
            linewidth=2.2, label="Training Accuracy")
    ax.plot(epochs, [a * 100 for a in res_hist["val_acc"]], "s--", color=PALETTE["secondary"],
            linewidth=2.2, label="Validation Accuracy")
    ax.axvline(5, color="gray", linestyle=":", label="Best Epoch (Epoch 5 - 80.35%)")
    ax.set_title("Pretrained ResNet-18: Training vs. Validation Accuracy", pad=15)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 100)
    ax.set_xticks(epochs)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_03_resnet18_accuracy", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 4: ResNet-18 Training vs Validation Loss Curve
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, res_hist["train_loss"], "o-", color=PALETTE["accent"],
            linewidth=2.2, label="Training Cross-Entropy Loss")
    ax.plot(epochs, res_hist["val_loss"], "s--", color=PALETTE["warning"],
            linewidth=2.2, label="Validation Loss")
    ax.axvline(5, color="gray", linestyle=":", label="Best Checkpoint (Epoch 5)")
    ax.set_title("Pretrained ResNet-18: Training vs. Validation Loss Dynamics", pad=15)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Loss")
    ax.set_xticks(epochs)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_04_resnet18_loss", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 5: Custom CNN vs ResNet-18 Accuracy Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    models = ["Custom CNN", "Pretrained ResNet-18"]
    acc_vals = [59.95, 80.35]
    colors = [PALETTE["primary"], PALETTE["secondary"]]

    bars = ax.bar(models, acc_vals, color=colors, width=0.45, edgecolor="black", linewidth=1)
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Validation Accuracy Comparison: Custom CNN vs. ResNet-18", pad=15)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_05_dl_accuracy_comparison", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 6: Custom CNN vs ResNet-18 ROC-AUC Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    auc_vals = [0.7616, 0.7507]
    bars = ax.bar(models, auc_vals, color=[PALETTE["teal"], PALETTE["warning"]], width=0.45, edgecolor="black", linewidth=1)
    for bar, val in zip(bars, auc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Validation ROC-AUC Comparison: Custom CNN vs. ResNet-18", pad=15)
    ax.set_ylabel("ROC-AUC Score")
    ax.set_ylim(0.4, 0.90)
    ax.axhline(0.50, color="gray", linestyle=":", label="Chance (0.50)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_06_dl_roc_auc_comparison", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 7: Custom CNN vs ResNet-18 Sensitivity (Recall) Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rec_vals = [79.03, 40.32]
    bars = ax.bar(models, rec_vals, color=[PALETTE["anemia"], PALETTE["warning"]], width=0.45, edgecolor="black", linewidth=1)
    for bar, val in zip(bars, rec_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Validation Sensitivity (Recall for Anemia): Custom CNN vs. ResNet-18", pad=15)
    ax.set_ylabel("Sensitivity / Recall (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_07_dl_sensitivity_comparison", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 8: Custom CNN vs ResNet-18 Specificity Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    spec_vals = [56.42, 87.76]
    bars = ax.bar(models, spec_vals, color=[PALETTE["primary"], PALETTE["secondary"]], width=0.45, edgecolor="black", linewidth=1)
    for bar, val in zip(bars, spec_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Validation Specificity Comparison: Custom CNN vs. ResNet-18", pad=15)
    ax.set_ylabel("Specificity (True Negative Rate %)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_08_dl_specificity_comparison", OUT_DIR)

    # -------------------------------------------------------------
    # Evaluate Validation Set for Continuous ROC & PR Curves
    # -------------------------------------------------------------
    print("Evaluating validation dataloader for continuous deep learning prediction scores...")
    _, val_loader, _, _ = create_dataloaders(
        train_manifest=PHASE2_DIR / "train.csv",
        val_manifest=PHASE2_DIR / "validation.csv",
        modality="Eye",
        batch_size=32,
        num_workers=0
    )

    cnn_model = get_custom_cnn(num_classes=2)
    _, _, _, cnn_probs = evaluate_checkpoint(cnn_model, PHASE4_DIR / "checkpoints" / "best_custom_cnn.pth", val_loader)

    res_model = get_resnet18_transfer_model(num_classes=2, freeze_backbone=True)
    _, y_val_dl, _, res_probs = evaluate_checkpoint(res_model, PHASE4_DIR / "checkpoints" / "best_resnet18.pth", val_loader)

    # -------------------------------------------------------------
    # Graph 9: Custom CNN vs ResNet-18 Validation ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fpr_cnn, tpr_cnn, _ = roc_curve(y_val_dl, cnn_probs)
    auc_cnn = auc(fpr_cnn, tpr_cnn)

    fpr_res, tpr_res, _ = roc_curve(y_val_dl, res_probs)
    auc_res = auc(fpr_res, tpr_res)

    ax.plot(fpr_cnn, tpr_cnn, label=f"Custom CNN (Validation AUC = {auc_cnn:.4f})",
            color=PALETTE["primary"], linewidth=2.2)
    ax.plot(fpr_res, tpr_res, label=f"Pretrained ResNet-18 (Validation AUC = {auc_res:.4f})",
            color=PALETTE["secondary"], linewidth=2.2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Chance (AUC = 0.5000)")

    ax.set_title("Validation ROC Curves: Custom CNN vs. Pretrained ResNet-18", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_09_dl_roc_curves", OUT_DIR)

    # -------------------------------------------------------------
    # Graph 10: Custom CNN vs ResNet-18 Validation PR Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    prec_cnn, rec_cnn, _ = precision_recall_curve(y_val_dl, cnn_probs)
    pr_auc_cnn = auc(rec_cnn, prec_cnn)

    prec_res, rec_res, _ = precision_recall_curve(y_val_dl, res_probs)
    pr_auc_res = auc(rec_res, prec_res)

    prev = np.mean(y_val_dl)

    ax.plot(rec_cnn, prec_cnn, label=f"Custom CNN (PR-AUC = {pr_auc_cnn:.4f})",
            color=PALETTE["primary"], linewidth=2.2)
    ax.plot(rec_res, prec_res, label=f"Pretrained ResNet-18 (PR-AUC = {pr_auc_res:.4f})",
            color=PALETTE["secondary"], linewidth=2.2)
    ax.axhline(prev, color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline Prevalence ({prev*100:.1f}%)")

    ax.set_title("Validation Precision-Recall Curves: Custom CNN vs. ResNet-18", pad=15)
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    save_academic_figure(fig, "P4_10_dl_precision_recall_curves", OUT_DIR)

    print("Phase 4 graphs successfully generated!")


if __name__ == "__main__":
    generate_phase4_graphs()
