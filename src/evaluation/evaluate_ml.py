"""
Evaluation Metrics and Diagnostic Reporting Module for Classical ML
Computes accuracy, precision, recall, F1, balanced accuracy, ROC-AUC, PR-AUC, and confusion matrices.
"""

import os
import json
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
CM_DIR = WORKSPACE_ROOT / "results" / "phase3" / "confusion_matrices"
CHARTS_DIR = WORKSPACE_ROOT / "results" / "phase3" / "charts"
REPORTS_DIR = WORKSPACE_ROOT / "results" / "phase3" / "reports"

CM_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray,
                   model_name: str, modality_name: str) -> dict:
    """
    Evaluates a trained classifier on validation features and returns comprehensive clinical metrics.
    """
    y_pred = model.predict(X_val)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_val)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        y_prob = y_pred.astype(float)

    # Calculate metrics
    acc = accuracy_score(y_val, y_pred)
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    prec_anemia = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
    rec_anemia = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1_anemia = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_val, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_val, y_prob)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_val, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / max(tn + fp, 1)
    else:
        tn = fp = fn = tp = 0
        specificity = 0.0

    # Save Confusion Matrix Plot
    safe_name = f"{modality_name}_{model_name}".lower().replace(" ", "_").replace("(", "").replace(")", "")
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

    # Classification report
    report_dict = classification_report(
        y_val, y_pred,
        target_names=["Non_Anemia", "Anemia"],
        output_dict=True,
        zero_division=0
    )
    report_txt = classification_report(
        y_val, y_pred,
        target_names=["Non_Anemia", "Anemia"],
        zero_division=0
    )

    # Save individual classification report text
    rep_txt_path = REPORTS_DIR / f"report_{safe_name}.txt"
    rep_txt_path.write_text(report_txt, encoding="utf-8")

    # ROC and PR Curve data points
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    prec_pts, rec_pts, _ = precision_recall_curve(y_val, y_prob)

    return {
        "model_name": model_name,
        "modality": modality_name,
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
        "confusion_matrix_path": str(cm_path),
        "classification_report_path": str(rep_txt_path),
        "curve_data": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision_pts": prec_pts.tolist(),
            "recall_pts": rec_pts.tolist()
        }
    }
