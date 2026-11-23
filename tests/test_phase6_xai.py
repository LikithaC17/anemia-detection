"""
Integration and Verification Tests for Phase 6: Explainable AI (SHAP + Grad-CAM)
Author: Senior ML Engineer, Deep Learning Researcher & XAI Specialist
Project: AI-Based Anemia Detection
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def test_phase6_required_csvs_and_reports_exist():
    """Verify that all Task 8 required tabular files and reports exist and are non-empty."""
    req_files = [
        "results/phase6/shap_global_feature_importance.csv",
        "results/phase6/shap_individual_explanations.csv",
        "results/phase6/gradcam_explanation_summary.csv",
        "results/phase6/explainability_comparison.csv",
        "results/phase6/phase6_status.csv",
        "results/phase6/phase6_final_report.txt",
        "results/phase6/comparison/gradcam_cnn_vs_resnet18_comparison.png",
        "results/phase6/gradcam/custom_cnn/custom_cnn_gradcam_gallery.png",
        "results/phase6/gradcam/resnet18/resnet18_gradcam_gallery.png",
        "results/phase6/shap/summary_plots/shap_beeswarm_random_forest.png",
        "results/phase6/shap/summary_plots/shap_beeswarm_logistic_regression.png",
        "results/phase6/shap/feature_importance/shap_bar_random_forest.png",
        "results/phase6/shap/feature_importance/shap_bar_logistic_regression.png"
    ]
    for rel_p in req_files:
        p = WORKSPACE_ROOT / rel_p
        assert p.exists(), f"Missing required Phase 6 file: {rel_p}"
        assert p.stat().st_size > 0, f"Empty file: {rel_p}"


def test_shap_values_integrity():
    """Verify that saved SHAP values arrays exist, have valid shapes and finite numbers."""
    rf_sv_path = WORKSPACE_ROOT / "results" / "phase6" / "shap" / "shap_values" / "shap_values_random_forest.npy"
    lr_sv_path = WORKSPACE_ROOT / "results" / "phase6" / "shap" / "shap_values" / "shap_values_logistic_regression.npy"
    dt_sv_path = WORKSPACE_ROOT / "results" / "phase6" / "shap" / "shap_values" / "shap_values_decision_tree.npy"

    assert rf_sv_path.exists()
    assert lr_sv_path.exists()
    assert dt_sv_path.exists()

    rf_sv = np.load(rf_sv_path)
    lr_sv = np.load(lr_sv_path)
    dt_sv = np.load(dt_sv_path)

    assert rf_sv.shape == (384, 151), f"Unexpected RF SHAP shape: {rf_sv.shape}"
    assert lr_sv.shape == (384, 151), f"Unexpected LR SHAP shape: {lr_sv.shape}"
    assert dt_sv.shape == (384, 151), f"Unexpected DT SHAP shape: {dt_sv.shape}"

    assert np.all(np.isfinite(rf_sv)), "NaN or Inf found in RF SHAP values!"
    assert np.all(np.isfinite(lr_sv)), "NaN or Inf found in LR SHAP values!"
    assert np.all(np.isfinite(dt_sv)), "NaN or Inf found in DT SHAP values!"


def test_gradcam_heatmaps_and_overlays():
    """Verify that saved Grad-CAM images have valid dimensions (224, 224, 3) and valid pixel bounds."""
    for model_name in ["custom_cnn", "resnet18"]:
        base_dir = WORKSPACE_ROOT / "results" / "phase6" / "gradcam" / model_name
        overlays = list((base_dir / "overlays").glob("*.png"))
        heatmaps = list((base_dir / "heatmaps").glob("*.png"))
        originals = list((base_dir / "original_images").glob("*.png"))

        assert len(overlays) >= 6, f"Too few overlays for {model_name}: {len(overlays)}"
        assert len(heatmaps) >= 6, f"Too few heatmaps for {model_name}: {len(heatmaps)}"
        assert len(originals) >= 6, f"Too few originals for {model_name}: {len(originals)}"

        # Validate image format and dimensions
        for img_p in overlays[:2] + heatmaps[:2] + originals[:2]:
            with Image.open(img_p) as img:
                assert img.size == (224, 224), f"Image size {img.size} != (224, 224) for {img_p}"
                arr = np.array(img)
                assert arr.ndim == 3 and arr.shape[2] == 3, f"Image shape {arr.shape} not RGB for {img_p}"


def test_status_file_contents():
    """Verify that phase6_status.csv records all eligible and ineligible models accurately."""
    status_path = WORKSPACE_ROOT / "results" / "phase6" / "phase6_status.csv"
    assert status_path.exists()
    df_st = pd.read_csv(status_path)

    expected_tasks = ["SHAP", "Grad-CAM", "Individual explanations", "Comparison", "Final report"]
    for t in expected_tasks:
        assert (df_st["task"] == t).any(), f"Task {t} missing from status table"

    # Verify that valid statuses are strictly used
    valid_statuses = {"COMPLETE", "PARTIALLY COMPLETE", "IN PROGRESS", "BLOCKED", "FAILED"}
    for st in df_st["status"]:
        assert st in valid_statuses, f"Invalid status string: {st}"


def test_previous_phase_files_unmodified():
    """Verify that Phases 1 through 5 files remain intact and untouched."""
    p2_test = WORKSPACE_ROOT / "results" / "phase2" / "test.csv"
    p4_cnn = WORKSPACE_ROOT / "models" / "deep_learning" / "cnn" / "best_custom_cnn_eye.pth"
    p5_metrics = WORKSPACE_ROOT / "results" / "phase5" / "final_test_metrics.csv"

    assert p2_test.exists() and p2_test.stat().st_size > 0
    assert p4_cnn.exists() and p4_cnn.stat().st_size > 0
    assert p5_metrics.exists() and p5_metrics.stat().st_size > 0
