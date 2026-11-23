"""
SHAP (SHapley Additive exPlanations) Module for Classical Machine Learning
Implements TreeExplainer and LinearExplainer for Random Forest, Logistic Regression, and Decision Tree.
Generates global feature rankings, beeswarm distributions, bar plots, and individual waterfall plots.

Author: Senior Machine Learning Engineer & Explainable AI Specialist
Project: AI-Based Anemia Detection
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
import shap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_shap_analysis(models_dir: Path,
                      train_features_path: Path,
                      test_features_path: Path,
                      feature_names_path: Path,
                      output_base_dir: Path,
                      selected_samples: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes comprehensive SHAP explanations for verified classical models on Eye Conjunctiva.
    """
    print("\n" + "=" * 80)
    print("RUNNING SHAP EXPLANATIONS FOR CLASSICAL MACHINE LEARNING MODELS")
    print("=" * 80)

    # 1. Load feature names and matrices
    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)["feature_names"]

    train_df = pd.read_csv(train_features_path)
    test_df = pd.read_csv(test_features_path)

    X_train = train_df[feature_names].values
    y_train = train_df["target_binary"].values
    X_test = test_df[feature_names].values
    y_test = test_df["target_binary"].values

    # Output subdirectories
    summary_dir = output_base_dir / "summary_plots"
    feat_imp_dir = output_base_dir / "feature_importance"
    indiv_dir = output_base_dir / "individual_explanations"
    values_dir = output_base_dir / "shap_values"
    for d in [summary_dir, feat_imp_dir, indiv_dir, values_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Background distribution: 100 representative samples from train set
    np.random.seed(42)
    bg_idx = np.random.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    X_bg = X_train[bg_idx]

    global_importance_rows = []
    individual_rows = []

    # -------------------------------------------------------------
    # Model 1: Random Forest (TreeExplainer)
    # -------------------------------------------------------------
    rf_path = models_dir / "random_forest_eye_conjunctiva.joblib"
    if rf_path.exists():
        print("Computing SHAP values for Random Forest (TreeExplainer)...")
        rf_model = joblib.load(rf_path)
        explainer_rf = shap.TreeExplainer(rf_model)
        sv_rf_all = explainer_rf.shap_values(X_test)

        # sv_rf_all shape: (N, 151, 2)
        if isinstance(sv_rf_all, np.ndarray) and sv_rf_all.ndim == 3:
            sv_rf_anemia = sv_rf_all[:, :, 1]
            ev_rf = explainer_rf.expected_value[1]
        elif isinstance(sv_rf_all, list):
            sv_rf_anemia = sv_rf_all[1]
            ev_rf = explainer_rf.expected_value[1]
        else:
            sv_rf_anemia = sv_rf_all
            ev_rf = explainer_rf.expected_value

        # Save SHAP values array
        np.save(values_dir / "shap_values_random_forest.npy", sv_rf_anemia)

        # Global feature importance (Mean Absolute SHAP Value)
        mean_abs_rf = np.mean(np.abs(sv_rf_anemia), axis=0)
        top_idx_rf = np.argsort(mean_abs_rf)[::-1]
        for rank, idx in enumerate(top_idx_rf[:30], 1):
            feat_name = feature_names[idx]
            feat_type = "Color" if any(k in feat_name.lower() for k in ["hist", "mean", "std", "rgb", "hsv"]) else "Texture"
            global_importance_rows.append({
                "model_name": "Random Forest",
                "modality": "Eye Conjunctiva",
                "rank": rank,
                "feature_name": feat_name,
                "feature_type": feat_type,
                "mean_abs_shap_value": round(float(mean_abs_rf[idx]), 6),
                "shap_explainer_type": "TreeExplainer"
            })

        # Summary Bar Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv_rf_anemia, X_test, feature_names=feature_names, plot_type="bar",
                          show=False, max_display=15)
        plt.title("Random Forest (Eye Conjunctiva) - Top 15 SHAP Global Features", fontsize=11, fontweight="bold")
        plt.tight_layout()
        rf_bar_p = feat_imp_dir / "shap_bar_random_forest.png"
        plt.savefig(rf_bar_p, dpi=200, bbox_inches="tight")
        plt.close()

        # Summary Beeswarm Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv_rf_anemia, X_test, feature_names=feature_names, show=False, max_display=15)
        plt.title("Random Forest (Eye Conjunctiva) - SHAP Beeswarm Summary Plot", fontsize=11, fontweight="bold")
        plt.tight_layout()
        rf_bee_p = summary_dir / "shap_beeswarm_random_forest.png"
        plt.savefig(rf_bee_p, dpi=200, bbox_inches="tight")
        plt.close()

        # Individual explanations for selected samples
        for s in selected_samples:
            s_idx = s["index"]
            if s_idx >= len(X_test):
                continue
            x_sample = X_test[s_idx]
            sv_sample = sv_rf_anemia[s_idx]
            prob = rf_model.predict_proba(x_sample.reshape(1, -1))[0, 1]

            # Top 3 positive and negative features for this sample
            top_pos = np.argsort(sv_sample)[::-1][:3]
            top_neg = np.argsort(sv_sample)[:3]

            pos_desc = "; ".join([f"{feature_names[j]} (+{sv_sample[j]:.3f})" for j in top_pos if sv_sample[j] > 0])
            neg_desc = "; ".join([f"{feature_names[j]} ({sv_sample[j]:.3f})" for j in top_neg if sv_sample[j] < 0])

            # Generate individual waterfall plot
            exp_obj = shap.Explanation(
                values=sv_sample,
                base_values=ev_rf,
                data=x_sample,
                feature_names=feature_names
            )
            plt.figure(figsize=(9, 5))
            shap.plots.waterfall(exp_obj, max_display=10, show=False)
            plt.title(f"RF Explanation: {s['category']}\nTrue: {s['true_label']} | Pred Prob(Anemia): {prob:.2f}",
                      fontsize=10, fontweight="bold")
            plt.tight_layout()
            wf_path = indiv_dir / f"waterfall_rf_sample_{s_idx:03d}.png"
            plt.savefig(wf_path, dpi=200, bbox_inches="tight")
            plt.close()

            individual_rows.append({
                "model_name": "Random Forest",
                "modality": "Eye Conjunctiva",
                "sample_index": s_idx,
                "category": s["category"],
                "subject_id": s.get("subject_id", "Unknown"),
                "image_path": s["image_path"],
                "true_label": s["true_label"],
                "predicted_label": "Anemia" if prob >= 0.5 else "Non_Anemia",
                "predicted_prob_anemia": round(float(prob), 4),
                "top_anemia_promoting_features": pos_desc or "None",
                "top_anemia_inhibiting_features": neg_desc or "None",
                "waterfall_plot_path": str(wf_path)
            })

    # -------------------------------------------------------------
    # Model 2: Logistic Regression (LinearExplainer)
    # -------------------------------------------------------------
    lr_path = models_dir / "logistic_regression_eye_conjunctiva.joblib"
    if lr_path.exists():
        print("Computing SHAP values for Logistic Regression (LinearExplainer)...")
        lr_pipe = joblib.load(lr_path)
        scaler = lr_pipe.named_steps["scaler"]
        lr_clf = lr_pipe.named_steps["lr"]

        X_bg_scaled = scaler.transform(X_bg)
        X_test_scaled = scaler.transform(X_test)

        explainer_lr = shap.LinearExplainer(lr_clf, X_bg_scaled)
        sv_lr = explainer_lr.shap_values(X_test_scaled)
        ev_lr = explainer_lr.expected_value

        np.save(values_dir / "shap_values_logistic_regression.npy", sv_lr)

        # Global feature importance
        mean_abs_lr = np.mean(np.abs(sv_lr), axis=0)
        top_idx_lr = np.argsort(mean_abs_lr)[::-1]
        for rank, idx in enumerate(top_idx_lr[:30], 1):
            feat_name = feature_names[idx]
            feat_type = "Color" if any(k in feat_name.lower() for k in ["hist", "mean", "std", "rgb", "hsv"]) else "Texture"
            global_importance_rows.append({
                "model_name": "Logistic Regression",
                "modality": "Eye Conjunctiva",
                "rank": rank,
                "feature_name": feat_name,
                "feature_type": feat_type,
                "mean_abs_shap_value": round(float(mean_abs_lr[idx]), 6),
                "shap_explainer_type": "LinearExplainer"
            })

        # Summary Bar Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv_lr, X_test_scaled, feature_names=feature_names, plot_type="bar",
                          show=False, max_display=15)
        plt.title("Logistic Regression (Eye Conjunctiva) - Top 15 SHAP Global Features", fontsize=11, fontweight="bold")
        plt.tight_layout()
        lr_bar_p = feat_imp_dir / "shap_bar_logistic_regression.png"
        plt.savefig(lr_bar_p, dpi=200, bbox_inches="tight")
        plt.close()

        # Summary Beeswarm Plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv_lr, X_test_scaled, feature_names=feature_names, show=False, max_display=15)
        plt.title("Logistic Regression (Eye Conjunctiva) - SHAP Beeswarm Summary Plot", fontsize=11, fontweight="bold")
        plt.tight_layout()
        lr_bee_p = summary_dir / "shap_beeswarm_logistic_regression.png"
        plt.savefig(lr_bee_p, dpi=200, bbox_inches="tight")
        plt.close()

        # Individual explanations for selected samples
        for s in selected_samples:
            s_idx = s["index"]
            if s_idx >= len(X_test):
                continue
            x_sample_scaled = X_test_scaled[s_idx]
            sv_sample = sv_lr[s_idx]
            prob = lr_pipe.predict_proba(X_test[s_idx].reshape(1, -1))[0, 1]

            top_pos = np.argsort(sv_sample)[::-1][:3]
            top_neg = np.argsort(sv_sample)[:3]

            pos_desc = "; ".join([f"{feature_names[j]} (+{sv_sample[j]:.3f})" for j in top_pos if sv_sample[j] > 0])
            neg_desc = "; ".join([f"{feature_names[j]} ({sv_sample[j]:.3f})" for j in top_neg if sv_sample[j] < 0])

            exp_obj = shap.Explanation(
                values=sv_sample,
                base_values=ev_lr,
                data=x_sample_scaled,
                feature_names=feature_names
            )
            plt.figure(figsize=(9, 5))
            shap.plots.waterfall(exp_obj, max_display=10, show=False)
            plt.title(f"Logistic Reg Explanation: {s['category']}\nTrue: {s['true_label']} | Pred Prob(Anemia): {prob:.2f}",
                      fontsize=10, fontweight="bold")
            plt.tight_layout()
            wf_path = indiv_dir / f"waterfall_lr_sample_{s_idx:03d}.png"
            plt.savefig(wf_path, dpi=200, bbox_inches="tight")
            plt.close()

            individual_rows.append({
                "model_name": "Logistic Regression",
                "modality": "Eye Conjunctiva",
                "sample_index": s_idx,
                "category": s["category"],
                "subject_id": s.get("subject_id", "Unknown"),
                "image_path": s["image_path"],
                "true_label": s["true_label"],
                "predicted_label": "Anemia" if prob >= 0.5 else "Non_Anemia",
                "predicted_prob_anemia": round(float(prob), 4),
                "top_anemia_promoting_features": pos_desc or "None",
                "top_anemia_inhibiting_features": neg_desc or "None",
                "waterfall_plot_path": str(wf_path)
            })

    # -------------------------------------------------------------
    # Model 3: Decision Tree (TreeExplainer)
    # -------------------------------------------------------------
    dt_path = models_dir / "decision_tree_eye_conjunctiva.joblib"
    if dt_path.exists():
        print("Computing SHAP values for Decision Tree (TreeExplainer)...")
        dt_model = joblib.load(dt_path)
        explainer_dt = shap.TreeExplainer(dt_model)
        sv_dt_all = explainer_dt.shap_values(X_test)
        if isinstance(sv_dt_all, np.ndarray) and sv_dt_all.ndim == 3:
            sv_dt_anemia = sv_dt_all[:, :, 1]
        elif isinstance(sv_dt_all, list):
            sv_dt_anemia = sv_dt_all[1]
        else:
            sv_dt_anemia = sv_dt_all

        np.save(values_dir / "shap_values_decision_tree.npy", sv_dt_anemia)

        mean_abs_dt = np.mean(np.abs(sv_dt_anemia), axis=0)
        top_idx_dt = np.argsort(mean_abs_dt)[::-1]
        for rank, idx in enumerate(top_idx_dt[:20], 1):
            feat_name = feature_names[idx]
            feat_type = "Color" if any(k in feat_name.lower() for k in ["hist", "mean", "std", "rgb", "hsv"]) else "Texture"
            global_importance_rows.append({
                "model_name": "Decision Tree",
                "modality": "Eye Conjunctiva",
                "rank": rank,
                "feature_name": feat_name,
                "feature_type": feat_type,
                "mean_abs_shap_value": round(float(mean_abs_dt[idx]), 6),
                "shap_explainer_type": "TreeExplainer"
            })

    df_global = pd.DataFrame(global_importance_rows)
    df_indiv = pd.DataFrame(individual_rows)

    print("SHAP explanations completed successfully!")
    return df_global, df_indiv
