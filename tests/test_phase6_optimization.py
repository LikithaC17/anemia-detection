"""
Automated Test Suite for Phase 6 Limitation Improvements
Verifies SHAP optimization for SVM & KNN, LayerCAM, Guided Backprop, Guided Grad-CAM,
and all academic comparison reports and tables.

Author: Senior Machine Learning Engineer & QA Specialist
Project: AI-Based Anemia Detection
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
OPT_BASE = WORKSPACE_ROOT / "results" / "phase6" / "optimization"
SHAP_OPT = OPT_BASE / "shap"
GRAD_OPT = OPT_BASE / "gradcam"
REPORTS_OPT = OPT_BASE / "reports"


def test_optimization_directories_exist():
    assert OPT_BASE.exists(), f"Optimization base directory missing: {OPT_BASE}"
    assert SHAP_OPT.exists(), f"SHAP optimization dir missing: {SHAP_OPT}"
    assert GRAD_OPT.exists(), f"Grad-CAM optimization dir missing: {GRAD_OPT}"
    assert (SHAP_OPT / "svm").exists()
    assert (SHAP_OPT / "knn").exists()
    assert (GRAD_OPT / "custom_cnn").exists()
    assert (GRAD_OPT / "resnet18").exists()


def test_svm_shap_outputs():
    svm_dir = SHAP_OPT / "svm"
    npy_path = svm_dir / "shap_values_svm.npy"
    bar_path = svm_dir / "shap_bar_svm.png"
    beeswarm_path = svm_dir / "shap_beeswarm_svm.png"
    importance_csv = svm_dir / "svm_global_feature_importance.csv"
    indiv_csv = svm_dir / "svm_individual_explanations.csv"

    assert npy_path.exists(), "svm shap_values_svm.npy missing"
    assert bar_path.exists(), "svm shap_bar_svm.png missing"
    assert beeswarm_path.exists(), "svm shap_beeswarm_svm.png missing"
    assert importance_csv.exists(), "svm_global_feature_importance.csv missing"
    assert indiv_csv.exists(), "svm_individual_explanations.csv missing"

    # Value checks
    arr = np.load(npy_path)
    assert arr.ndim == 2, f"Expected 2D array, got {arr.shape}"
    assert arr.shape[1] == 151, f"Expected 151 features, got {arr.shape[1]}"
    assert np.all(np.isfinite(arr)), "SVM SHAP array contains NaN or Inf"
    assert np.any(arr != 0), "SVM SHAP values are all zeros"

    df_imp = pd.read_csv(importance_csv)
    assert len(df_imp) > 0, "SVM feature importance is empty"
    assert "mean_abs_shap_value" in df_imp.columns

    df_indiv = pd.read_csv(indiv_csv)
    assert len(df_indiv) > 0, "SVM individual explanations are empty"


def test_knn_shap_outputs():
    knn_dir = SHAP_OPT / "knn"
    npy_path = knn_dir / "shap_values_knn.npy"
    bar_path = knn_dir / "shap_bar_knn.png"
    beeswarm_path = knn_dir / "shap_beeswarm_knn.png"
    importance_csv = knn_dir / "knn_global_feature_importance.csv"
    indiv_csv = knn_dir / "knn_individual_explanations.csv"

    assert npy_path.exists(), "knn shap_values_knn.npy missing"
    assert bar_path.exists(), "knn shap_bar_knn.png missing"
    assert beeswarm_path.exists(), "knn shap_beeswarm_knn.png missing"
    assert importance_csv.exists(), "knn_global_feature_importance.csv missing"
    assert indiv_csv.exists(), "knn_individual_explanations.csv missing"

    # Value checks
    arr = np.load(npy_path)
    assert arr.ndim == 2, f"Expected 2D array, got {arr.shape}"
    assert arr.shape[1] == 151, f"Expected 151 features, got {arr.shape[1]}"
    assert np.all(np.isfinite(arr)), "KNN SHAP array contains NaN or Inf"
    assert np.any(arr != 0), "KNN SHAP values are all zeros"

    df_imp = pd.read_csv(importance_csv)
    assert len(df_imp) > 0, "KNN feature importance is empty"

    df_indiv = pd.read_csv(indiv_csv)
    assert len(df_indiv) > 0, "KNN individual explanations are empty"


def test_enhanced_cam_and_guided_backprop_custom_cnn():
    cnn_dir = GRAD_OPT / "custom_cnn"
    for subdir in ["gradcam", "layercam", "guided_backprop", "guided_gradcam", "comparisons"]:
        d = cnn_dir / subdir
        assert d.exists(), f"Subdirectory {subdir} missing in custom_cnn"
        pngs = list(d.glob("*.png"))
        assert len(pngs) > 0, f"No PNG images found in {d}"

    comp_img = cnn_dir / "comparisons" / "custom_cnn_method_comparison.png"
    assert comp_img.exists(), "Custom CNN method comparison figure missing"
    with Image.open(comp_img) as img:
        assert img.size[0] > 500 and img.size[1] > 500, "Comparison figure dimensions too small"


def test_enhanced_cam_and_guided_backprop_resnet18():
    resnet_dir = GRAD_OPT / "resnet18"
    for subdir in ["gradcam", "layercam", "guided_backprop", "guided_gradcam", "comparisons"]:
        d = resnet_dir / subdir
        assert d.exists(), f"Subdirectory {subdir} missing in resnet18"
        pngs = list(d.glob("*.png"))
        assert len(pngs) > 0, f"No PNG images found in {d}"

    comp_img = resnet_dir / "comparisons" / "resnet18_method_comparison.png"
    assert comp_img.exists(), "ResNet-18 method comparison figure missing"
    with Image.open(comp_img) as img:
        assert img.size[0] > 500 and img.size[1] > 500, "Comparison figure dimensions too small"


def test_comparison_csv_tables():
    runtime_csv = OPT_BASE / "shap_runtime_comparison.csv"
    res_csv = OPT_BASE / "gradcam_resolution_comparison.csv"
    method_csv = OPT_BASE / "gradcam_method_comparison.csv"
    status_csv = OPT_BASE / "phase6_optimization_status.csv"

    for p in [runtime_csv, res_csv, method_csv, status_csv]:
        assert p.exists(), f"Required table missing: {p}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"CSV is empty: {p}"

    df_status = pd.read_csv(status_csv)
    assert all(df_status["status"] == "COMPLETE"), "Not all optimization tasks marked COMPLETE"


def test_improvement_reports():
    svm_rep = OPT_BASE / "svm_shap_optimization_report.txt"
    knn_rep = OPT_BASE / "knn_shap_optimization_report.txt"
    phase6_rep = OPT_BASE / "phase6_limitation_improvement_report.txt"

    for rep in [svm_rep, knn_rep, phase6_rep]:
        assert rep.exists(), f"Report file missing: {rep}"
        content = rep.read_text(encoding="utf-8")
        assert len(content.strip()) > 200, f"Report content too short: {rep}"
        assert "DISCLAIMER" in content.upper() or "CAUSALITY" in content.upper(), "Medical research disclaimer missing"
