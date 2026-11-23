"""
Integration and Verification Tests for Phase 5: Final Model Evaluation & Comparison
Author: Senior ML Engineer, Deep Learning Researcher & Model Evaluation Specialist
Project: AI-Based Anemia Detection
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def test_phase5_artifacts_exist():
    """Verify that all Phase 5 required output files are present and non-empty."""
    req_files = [
        "results/phase5/final_test_metrics.csv",
        "results/phase5/model_comparison.csv",
        "results/phase5/class_wise_metrics.csv",
        "results/phase5/classification_reports.txt",
        "results/phase5/error_analysis.csv",
        "results/phase5/phase5_final_report.txt",
        "results/phase5/error_analysis/misclassified_samples.png",
        "results/phase5/charts/accuracy_comparison.png",
        "results/phase5/charts/balanced_accuracy_comparison.png",
        "results/phase5/charts/recall_comparison.png",
        "results/phase5/charts/precision_comparison.png",
        "results/phase5/charts/f1_score_comparison.png",
        "results/phase5/charts/roc_auc_comparison.png",
        "results/phase5/charts/master_performance_comparison.png",
        "results/phase5/roc_curves/roc_curves_eye_conjunctiva.png",
        "results/phase5/precision_recall_curves/pr_curves_eye_conjunctiva.png"
    ]
    for rel_path in req_files:
        p = WORKSPACE_ROOT / rel_path
        assert p.exists(), f"Missing required file: {rel_path}"
        assert p.stat().st_size > 0, f"File is empty: {rel_path}"


def test_model_count_and_modalities():
    """Verify that all 17 models are present in final_test_metrics.csv across Eye, Palm, and Nail."""
    csv_path = WORKSPACE_ROOT / "results" / "phase5" / "final_test_metrics.csv"
    df = pd.read_csv(csv_path)
    
    assert len(df) == 17, f"Expected 17 models, found {len(df)}"
    
    # Check modality counts
    eye_models = df[df["modality"] == "Eye Conjunctiva"]
    palm_models = df[df["modality"] == "Palmar Pallor"]
    nail_models = df[df["modality"] == "Nail Beds"]
    
    assert len(eye_models) == 7, f"Expected 7 Eye models (5 ML + 2 DL), found {len(eye_models)}"
    assert len(palm_models) == 5, f"Expected 5 Palm models, found {len(palm_models)}"
    assert len(nail_models) == 5, f"Expected 5 Nail models, found {len(nail_models)}"


def test_metric_validity_bounds():
    """Verify that all computed test metrics obey statistical and theoretical bounds."""
    csv_path = WORKSPACE_ROOT / "results" / "phase5" / "final_test_metrics.csv"
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        # Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1 in [0, 1]
        for col in ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1_score"]:
            val = row[col]
            assert 0.0 <= val <= 1.0, f"Metric {col} out of bounds: {val} for {row['model_name']} ({row['modality']})"
            
        # MCC in [-1, 1]
        mcc = row["mcc"]
        assert -1.0 <= mcc <= 1.0, f"MCC out of bounds: {mcc}"
        
        # Confusion matrix sum equals total test samples
        cm_sum = row["true_positives"] + row["false_positives"] + row["true_negatives"] + row["false_negatives"]
        assert cm_sum == row["total_test_samples"], f"CM sum {cm_sum} != total {row['total_test_samples']}"
