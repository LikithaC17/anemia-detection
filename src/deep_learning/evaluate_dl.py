"""
Deep Learning Evaluation and Diagnostic Reporting Module
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
CM_DIR = WORKSPACE_ROOT / "results" / "phase4" / "confusion_matrices"
CHARTS_DIR = WORKSPACE_ROOT / "results" / "phase4" / "charts"
REPORTS_DIR = WORKSPACE_ROOT / "results" / "phase4" / "reports"

CM_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_deep_learning_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    modality_name: str,
    history: dict,
    training_time: float = 0.0
) -> dict:
    """
    Computes diagnostic metrics, confusion matrix, and training curves.
    """
    safe_name = f"{modality_name}_{model_name}".lower().replace(" ", "_").replace("-", "")

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec_anemia = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_anemia = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_anemia = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / max(tn + fp, 1)
    else:
        tn = fp = fn = tp = 0
        specificity = 0.0

    # 1. Confusion Matrix Plot
    cm_path = CM_DIR / f"cm_{safe_name}.png"
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non_Anemia (0)", "Anemia (1)"],
                yticklabels=["Non_Anemia (0)", "Anemia (1)"])
    plt.title(f"{model_name} ({modality_name})\nValidation Confusion Matrix", fontsize=10, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=9)
    plt.ylabel("True Ground Truth", fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. Training and Validation Learning Curves Plot
    epochs = range(1, len(history["train_loss"]) + 1)
    curves_path = CHARTS_DIR / f"learning_curves_{safe_name}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c", lw=2)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#3498db", lw=2)
    ax1.set_title(f"{model_name} ({modality_name}) - Loss Curves", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ecc71", lw=2)
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Accuracy", color="#9b59b6", lw=2)
    ax2.plot(epochs, history["val_bal_acc"], "^--", label="Val Balanced Acc", color="#f39c12", lw=2)
    ax2.set_title(f"{model_name} ({modality_name}) - Accuracy Curves", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy Score", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(curves_path, dpi=200)
    plt.close()

    # 3. Classification Report Text
    report_txt = classification_report(
        y_true, y_pred,
        target_names=["Non_Anemia", "Anemia"],
        zero_division=0
    )
    rep_txt_path = REPORTS_DIR / f"report_{safe_name}.txt"
    rep_txt_path.write_text(report_txt, encoding="utf-8")

    # ROC and PR points
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, y_prob)

    return {
        "model_name": model_name,
        "modality": modality_name,
        "model_type": "Deep Learning",
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "precision_anemia": round(float(prec_anemia), 4),
        "recall_sensitivity": round(float(rec_anemia), 4),
        "specificity": round(float(specificity), 4),
        "f1_anemia": round(float(f1_anemia), 4),
        "f1_macro": round(float(f1_macro), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "training_time_seconds": round(float(training_time), 2),
        "confusion_matrix_path": str(cm_path),
        "curves_path": str(curves_path),
        "report_path": str(rep_txt_path),
        "curve_data": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision_pts": prec_pts.tolist(),
            "recall_pts": rec_pts.tolist()
        }
    }
