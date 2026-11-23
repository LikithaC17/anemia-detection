"""
Model Comparison and Diagnostic Visualization Module
Generates comparison CSVs, multi-model ROC/PR curves, and comparative bar charts.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
CHARTS_DIR = WORKSPACE_ROOT / "results" / "phase3" / "charts"
REPORTS_DIR = WORKSPACE_ROOT / "results" / "phase3" / "reports"


def generate_comparison_artifacts(eval_results: list[dict], output_prefix: str = "") -> pd.DataFrame:
    """
    Creates comparison tables and visual diagnostic charts for all evaluated models.
    """
    df_rows = []
    curves_dict = {}

    for res in eval_results:
        key = f"{res['modality']}_{res['model_name']}"
        curves_dict[key] = res.get("curve_data", {})
        row = {k: v for k, v in res.items() if k != "curve_data"}
        df_rows.append(row)

    comp_df = pd.DataFrame(df_rows)
    prefix_str = f"{output_prefix}_" if output_prefix else ""
    csv_path = REPORTS_DIR / f"{prefix_str}model_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to: {csv_path}")

    # 1. Accuracy & Balanced Accuracy Comparison Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(comp_df))
    width = 0.35

    plt.bar(x - width/2, comp_df["accuracy"], width, label="Standard Accuracy", color="#3498db", alpha=0.85, edgecolor="black")
    plt.bar(x + width/2, comp_df["balanced_accuracy"], width, label="Balanced Accuracy", color="#2ecc71", alpha=0.85, edgecolor="black")

    labels = [f"{r['model_name']}\n({r['modality']})" for _, r in comp_df.iterrows()]
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Score", fontsize=11)
    plt.title("Model Accuracy vs. Balanced Accuracy Comparison on Validation Set", fontsize=12, fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    acc_chart_path = CHARTS_DIR / f"{prefix_str}accuracy_comparison.png"
    plt.savefig(acc_chart_path, dpi=200)
    plt.close()

    # 2. Precision, Recall (Sensitivity), and F1 Comparison Chart
    plt.figure(figsize=(14, 6))
    width = 0.25
    plt.bar(x - width, comp_df["precision_anemia"], width, label="Precision (Anemia)", color="#e67e22", alpha=0.85, edgecolor="black")
    plt.bar(x, comp_df["recall_sensitivity"], width, label="Recall / Sensitivity (Anemia)", color="#e74c3c", alpha=0.85, edgecolor="black")
    plt.bar(x + width, comp_df["f1_macro"], width, label="Macro F1-Score", color="#9b59b6", alpha=0.85, edgecolor="black")

    plt.xticks(x, labels, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Score", fontsize=11)
    plt.title("Clinical Diagnostic Metrics (Precision, Sensitivity, Macro F1) Across Models", fontsize=12, fontweight="bold")
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    metrics_chart_path = CHARTS_DIR / f"{prefix_str}precision_recall_f1_comparison.png"
    plt.savefig(metrics_chart_path, dpi=200)
    plt.close()

    # 3. Multi-Model ROC Curves
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(comp_df))

    for idx, (_, r) in enumerate(comp_df.iterrows()):
        key = f"{r['modality']}_{r['model_name']}"
        cdata = curves_dict.get(key, {})
        fpr = cdata.get("fpr", [])
        tpr = cdata.get("tpr", [])
        if fpr and tpr:
            plt.plot(fpr, tpr, color=palette[idx], lw=2,
                     label=f"{r['model_name']} ({r['modality']}) - AUC: {r['roc_auc']:.3f}")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC: 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("ROC Curves Comparison on Validation Set", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_chart_path = CHARTS_DIR / f"{prefix_str}roc_curves.png"
    plt.savefig(roc_chart_path, dpi=200)
    plt.close()

    # 4. Multi-Model Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for idx, (_, r) in enumerate(comp_df.iterrows()):
        key = f"{r['modality']}_{r['model_name']}"
        cdata = curves_dict.get(key, {})
        prec = cdata.get("precision_pts", [])
        rec = cdata.get("recall_pts", [])
        if prec and rec:
            plt.plot(rec, prec, color=palette[idx], lw=2,
                     label=f"{r['model_name']} ({r['modality']}) - PR-AUC: {r['pr_auc']:.3f}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves Comparison on Validation Set", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pr_chart_path = CHARTS_DIR / f"{prefix_str}pr_curves.png"
    plt.savefig(pr_chart_path, dpi=200)
    plt.close()

    return comp_df
