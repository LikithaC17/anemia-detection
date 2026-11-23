"""
Optimized SHAP Engine for Non-Linear SVM (RBF) and KNN Classifiers
Solves the computational bottleneck via controlled background sampling and bounded evaluation budgets.

Author: Senior Machine Learning Engineer & Explainable AI Specialist
Project: AI-Based Anemia Detection
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_optimized_svm_shap(svm_model_path: Path,
                           X_train: np.ndarray,
                           X_test: np.ndarray,
                           feature_names: List[str],
                           selected_samples: List[Dict[str, Any]],
                           output_dir: Path,
                           n_bg: int = 30,
                           nsamples: int = 100) -> Dict[str, Any]:
    """
    Executes optimized KernelExplainer for non-linear RBF SVM on Anemia probability.
    """
    print(f"\n[OPTIMIZED SHAP] Running KernelExplainer for SVM (RBF Kernel, n_bg={n_bg}, nsamples={nsamples})...")
    output_dir.mkdir(parents=True, exist_ok=True)
    indiv_dir = output_dir / "individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    svm_pipe = joblib.load(svm_model_path)

    # Controlled background sampling from train set
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), size=min(n_bg, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    # Subset of test samples to explain (representative instances)
    test_indices = [s["index"] for s in selected_samples if s["index"] < len(X_test)]
    X_test_subset = X_test[test_indices]

    def predict_anemia_prob(x_mat):
        return svm_pipe.predict_proba(x_mat)[:, 1]

    t0 = time.time()
    explainer = shap.KernelExplainer(predict_anemia_prob, X_bg)
    sv_svm = explainer.shap_values(X_test_subset, nsamples=nsamples)
    runtime = time.time() - t0

    print(f"SVM Kernel SHAP completed in {runtime:.2f}s for {len(test_indices)} instances.")

    # Save SHAP array
    np.save(output_dir / "shap_values_svm.npy", sv_svm)

    # Global feature importance across explained samples
    mean_abs_svm = np.mean(np.abs(sv_svm), axis=0)
    top_idx = np.argsort(mean_abs_svm)[::-1]
    importance_records = []
    for rank, idx in enumerate(top_idx[:30], 1):
        f_name = feature_names[idx]
        f_type = "Color" if any(k in f_name.lower() for k in ["hist", "mean", "std", "rgb", "hsv"]) else "Texture"
        importance_records.append({
            "model_name": "SVM (RBF)",
            "modality": "Eye Conjunctiva",
            "rank": rank,
            "feature_name": f_name,
            "feature_type": f_type,
            "mean_abs_shap_value": round(float(mean_abs_svm[idx]), 6),
            "shap_explainer_type": f"KernelExplainer (n_bg={n_bg}, nsamples={nsamples})"
        })
    df_importance = pd.DataFrame(importance_records)
    df_importance.to_csv(output_dir / "svm_global_feature_importance.csv", index=False)

    # Bar plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv_svm, X_test_subset, feature_names=feature_names, plot_type="bar",
                      show=False, max_display=15)
    plt.title(f"Optimized SVM (RBF) - Top 15 SHAP Features (KernelExplainer, {runtime:.1f}s)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar_svm.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv_svm, X_test_subset, feature_names=feature_names, show=False, max_display=15)
    plt.title("Optimized SVM (RBF) - SHAP Beeswarm Distribution", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm_svm.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Individual waterfall / bar plots
    indiv_records = []
    for k, s in enumerate(selected_samples):
        if s["index"] not in test_indices:
            continue
        idx_in_subset = test_indices.index(s["index"])
        sv_sample = sv_svm[idx_in_subset]
        x_sample = X_test_subset[idx_in_subset]
        prob = predict_anemia_prob(x_sample.reshape(1, -1))[0]

        top_pos = np.argsort(sv_sample)[::-1][:3]
        top_neg = np.argsort(sv_sample)[:3]
        pos_desc = "; ".join([f"{feature_names[j]} (+{sv_sample[j]:.3f})" for j in top_pos if sv_sample[j] > 0])
        neg_desc = "; ".join([f"{feature_names[j]} ({sv_sample[j]:.3f})" for j in top_neg if sv_sample[j] < 0])

        exp_obj = shap.Explanation(
            values=sv_sample,
            base_values=explainer.expected_value,
            data=x_sample,
            feature_names=feature_names
        )
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(exp_obj, max_display=10, show=False)
        plt.title(f"SVM Explanation: {s['category']}\nTrue: {s['true_label']} | Pred Prob(Anemia): {prob:.2f}",
                  fontsize=10, fontweight="bold")
        plt.tight_layout()
        wf_p = indiv_dir / f"waterfall_svm_sample_{s['index']:03d}.png"
        plt.savefig(wf_p, dpi=200, bbox_inches="tight")
        plt.close()

        indiv_records.append({
            "model_name": "SVM (RBF)",
            "sample_index": s["index"],
            "category": s["category"],
            "subject_id": s.get("subject_id", "Unknown"),
            "true_label": s["true_label"],
            "predicted_prob_anemia": round(float(prob), 4),
            "top_positive_features": pos_desc,
            "top_negative_features": neg_desc,
            "waterfall_plot_path": str(wf_p)
        })

    df_indiv = pd.DataFrame(indiv_records)
    df_indiv.to_csv(output_dir / "svm_individual_explanations.csv", index=False)

    return {
        "model_name": "SVM (RBF)",
        "explainer_type": "KernelExplainer",
        "n_background": n_bg,
        "n_explained": len(test_indices),
        "evaluation_budget_per_sample": nsamples,
        "total_runtime_seconds": round(runtime, 2),
        "runtime_per_sample_seconds": round(runtime / max(len(test_indices), 1), 3),
        "output_shape": list(sv_svm.shape),
        "top_features": [r["feature_name"] for r in importance_records[:5]]
    }


def run_optimized_knn_shap(knn_model_path: Path,
                           X_train: np.ndarray,
                           X_test: np.ndarray,
                           feature_names: List[str],
                           selected_samples: List[Dict[str, Any]],
                           output_dir: Path,
                           n_bg: int = 30,
                           nsamples: int = 100) -> Dict[str, Any]:
    """
    Executes optimized KernelExplainer for KNN (distance-weighted) on Anemia probability.
    """
    print(f"\n[OPTIMIZED SHAP] Running KernelExplainer for KNN (n_neighbors=15, n_bg={n_bg}, nsamples={nsamples})...")
    output_dir.mkdir(parents=True, exist_ok=True)
    indiv_dir = output_dir / "individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    knn_pipe = joblib.load(knn_model_path)

    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), size=min(n_bg, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    test_indices = [s["index"] for s in selected_samples if s["index"] < len(X_test)]
    X_test_subset = X_test[test_indices]

    def predict_anemia_prob(x_mat):
        return knn_pipe.predict_proba(x_mat)[:, 1]

    t0 = time.time()
    explainer = shap.KernelExplainer(predict_anemia_prob, X_bg)
    sv_knn = explainer.shap_values(X_test_subset, nsamples=nsamples)
    runtime = time.time() - t0

    print(f"KNN Kernel SHAP completed in {runtime:.2f}s for {len(test_indices)} instances.")

    np.save(output_dir / "shap_values_knn.npy", sv_knn)

    mean_abs_knn = np.mean(np.abs(sv_knn), axis=0)
    top_idx = np.argsort(mean_abs_knn)[::-1]
    importance_records = []
    for rank, idx in enumerate(top_idx[:30], 1):
        f_name = feature_names[idx]
        f_type = "Color" if any(k in f_name.lower() for k in ["hist", "mean", "std", "rgb", "hsv"]) else "Texture"
        importance_records.append({
            "model_name": "KNN (Distance-Weighted)",
            "modality": "Eye Conjunctiva",
            "rank": rank,
            "feature_name": f_name,
            "feature_type": f_type,
            "mean_abs_shap_value": round(float(mean_abs_knn[idx]), 6),
            "shap_explainer_type": f"KernelExplainer (n_bg={n_bg}, nsamples={nsamples})"
        })
    df_importance = pd.DataFrame(importance_records)
    df_importance.to_csv(output_dir / "knn_global_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv_knn, X_test_subset, feature_names=feature_names, plot_type="bar",
                      show=False, max_display=15)
    plt.title(f"Optimized KNN - Top 15 SHAP Features (KernelExplainer, {runtime:.1f}s)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar_knn.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv_knn, X_test_subset, feature_names=feature_names, show=False, max_display=15)
    plt.title("Optimized KNN - SHAP Beeswarm Distribution", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm_knn.png", dpi=200, bbox_inches="tight")
    plt.close()

    indiv_records = []
    for k, s in enumerate(selected_samples):
        if s["index"] not in test_indices:
            continue
        idx_in_subset = test_indices.index(s["index"])
        sv_sample = sv_knn[idx_in_subset]
        x_sample = X_test_subset[idx_in_subset]
        prob = predict_anemia_prob(x_sample.reshape(1, -1))[0]

        top_pos = np.argsort(sv_sample)[::-1][:3]
        top_neg = np.argsort(sv_sample)[:3]
        pos_desc = "; ".join([f"{feature_names[j]} (+{sv_sample[j]:.3f})" for j in top_pos if sv_sample[j] > 0])
        neg_desc = "; ".join([f"{feature_names[j]} ({sv_sample[j]:.3f})" for j in top_neg if sv_sample[j] < 0])

        exp_obj = shap.Explanation(
            values=sv_sample,
            base_values=explainer.expected_value,
            data=x_sample,
            feature_names=feature_names
        )
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(exp_obj, max_display=10, show=False)
        plt.title(f"KNN Explanation: {s['category']}\nTrue: {s['true_label']} | Pred Prob(Anemia): {prob:.2f}",
                  fontsize=10, fontweight="bold")
        plt.tight_layout()
        wf_p = indiv_dir / f"waterfall_knn_sample_{s['index']:03d}.png"
        plt.savefig(wf_p, dpi=200, bbox_inches="tight")
        plt.close()

        indiv_records.append({
            "model_name": "KNN",
            "sample_index": s["index"],
            "category": s["category"],
            "subject_id": s.get("subject_id", "Unknown"),
            "true_label": s["true_label"],
            "predicted_prob_anemia": round(float(prob), 4),
            "top_positive_features": pos_desc,
            "top_negative_features": neg_desc,
            "waterfall_plot_path": str(wf_p)
        })

    df_indiv = pd.DataFrame(indiv_records)
    df_indiv.to_csv(output_dir / "knn_individual_explanations.csv", index=False)

    return {
        "model_name": "KNN",
        "explainer_type": "KernelExplainer",
        "n_background": n_bg,
        "n_explained": len(test_indices),
        "evaluation_budget_per_sample": nsamples,
        "total_runtime_seconds": round(runtime, 2),
        "runtime_per_sample_seconds": round(runtime / max(len(test_indices), 1), 3),
        "output_shape": list(sv_knn.shape),
        "top_features": [r["feature_name"] for r in importance_records[:5]]
    }
