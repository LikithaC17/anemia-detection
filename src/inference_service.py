"""
Inference and Explainability Service for Flask Web Application
Manages model caching, preprocessing pipelines, model execution,
confidence scoring, and real-time XAI generation.

Author: Senior Machine Learning & Deep Learning Engineer
Project: AI-Based Anemia Detection
"""

import os
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root is in path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import Config
from src.deep_learning.custom_cnn import get_custom_cnn
from src.deep_learning.resnet18_model import get_resnet18_transfer_model
from src.deep_learning.transforms import get_val_transforms
from src.feature_extraction.color_features import extract_color_features
from src.feature_extraction.texture_features import extract_texture_features
from src.explainability.gradcam import GradCAM
from src.explainability.layercam import LayerCAM
from src.explainability.guided_backprop import GuidedBackprop
from src.explainability.explanation_utils import denormalize_tensor, apply_colormap_on_image

# Model Cache
_MODEL_CACHE: Dict[str, Any] = {}
_FEATURE_NAMES: Optional[list] = None
_SHAP_BACKGROUND: Optional[np.ndarray] = None


def get_feature_names() -> list:
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        fn_path = Config.PHASE3_RESULTS / "features" / "feature_names.json"
        with open(fn_path, "r", encoding="utf-8") as f:
            _FEATURE_NAMES = json.load(f)["feature_names"]
    return _FEATURE_NAMES


def get_shap_background() -> np.ndarray:
    global _SHAP_BACKGROUND
    if _SHAP_BACKGROUND is None:
        bg_path = Config.PHASE3_RESULTS / "features" / "eye_train_features.csv"
        df = pd.read_csv(bg_path)
        f_names = get_feature_names()
        X_all = df[f_names].values
        np.random.seed(42)
        idx = np.random.choice(len(X_all), size=min(30, len(X_all)), replace=False)
        _SHAP_BACKGROUND = X_all[idx]
    return _SHAP_BACKGROUND


def load_model(modality: str, model_id: str):
    cache_key = f"{modality}_{model_id}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if modality == "eye_conjunctiva" and model_id == "custom_cnn":
        model = get_custom_cnn(num_classes=2)
        chk_path = Config.MODELS_DIR / "deep_learning" / "cnn" / "best_custom_cnn_eye.pth"
        if not chk_path.exists():
            chk_path = Config.PHASE4_RESULTS / "checkpoints" / "best_custom_cnn.pth"
        chk = torch.load(chk_path, map_location="cpu")
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model

    if modality == "eye_conjunctiva" and model_id == "resnet18":
        model = get_resnet18_transfer_model(num_classes=2, freeze_backbone=True)
        chk_path = Config.MODELS_DIR / "deep_learning" / "resnet18" / "best_resnet18_eye.pth"
        if not chk_path.exists():
            chk_path = Config.PHASE4_RESULTS / "checkpoints" / "best_resnet18.pth"
        chk = torch.load(chk_path, map_location="cpu")
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
        _MODEL_CACHE[cache_key] = model
        return model

    # Classical ML models
    ml_filename = f"{model_id}_{modality}.joblib"
    ml_path = Config.CLASSICAL_ML_DIR / ml_filename
    if not ml_path.exists():
        raise FileNotFoundError(f"Classical ML model not found: {ml_path}")
    model = joblib.load(ml_path)
    _MODEL_CACHE[cache_key] = model
    return model


def extract_image_features(image_path: Path) -> Tuple[np.ndarray, list]:
    with Image.open(image_path) as pil_img:
        img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    color_feats, color_names = extract_color_features(img_rgb)
    texture_feats, texture_names = extract_texture_features(img_rgb)

    all_feats = np.concatenate([color_feats, texture_feats])
    all_names = color_names + texture_names
    return all_feats, all_names


def predict_classical_ml(model, features_vec: np.ndarray, model_id: str) -> Dict[str, Any]:
    X = features_vec.reshape(1, -1)
    pred_class_idx = int(model.predict(X)[0])
    
    # Calculate probability
    prob_anemia = None
    prob_non_anemia = None
    confidence = None
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_non_anemia = float(probs[0])
        prob_anemia = float(probs[1])
        confidence = prob_anemia if pred_class_idx == 1 else prob_non_anemia

    pred_label = "Anemia" if pred_class_idx == 1 else "Non_Anemia"

    return {
        "predicted_class": pred_label,
        "class_index": pred_class_idx,
        "prob_anemia": round(prob_anemia, 4) if prob_anemia is not None else None,
        "prob_non_anemia": round(prob_non_anemia, 4) if prob_non_anemia is not None else None,
        "confidence": round(confidence * 100, 2) if confidence is not None else None
    }


def predict_deep_learning(model, image_path: Path, model_id: str) -> Dict[str, Any]:
    with Image.open(image_path) as pil_img:
        img_rgb = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

    transforms_pipeline = get_val_transforms()
    input_tensor = transforms_pipeline(img_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class_idx = int(torch.argmax(probs).item())
        prob_non_anemia = float(probs[0].item())
        prob_anemia = float(probs[1].item())

    pred_label = "Anemia" if pred_class_idx == 1 else "Non_Anemia"
    confidence = prob_anemia if pred_class_idx == 1 else prob_non_anemia

    return {
        "predicted_class": pred_label,
        "class_index": pred_class_idx,
        "prob_anemia": round(prob_anemia, 4),
        "prob_non_anemia": round(prob_non_anemia, 4),
        "confidence": round(confidence * 100, 2)
    }


def explain_deep_learning(model, model_id: str, image_path: Path, output_subdir: Path, target_class: int) -> Dict[str, str]:
    """
    Generates Grad-CAM, LayerCAM, Guided Backprop, and Guided Grad-CAM for an uploaded image.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    transforms_pipeline = get_val_transforms()
    with Image.open(image_path) as pil_img:
        img_rgb = pil_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

    input_tensor = transforms_pipeline(img_rgb).unsqueeze(0).to(device)
    rgb_disp = denormalize_tensor(input_tensor)

    # Pick layers based on architecture
    if model_id == "custom_cnn":
        final_layer = model.block4[0]
        inter_layer = model.block3[0]
    else:  # resnet18
        final_layer = model.layer4[-1].conv2
        inter_layer = model.layer3[-1].conv2

    gradcam_engine = GradCAM(model, final_layer)
    layercam_engine = LayerCAM(model, final_layer)
    guided_engine = GuidedBackprop(model)

    try:
        # 1. Grad-CAM
        gcam_map, _, _ = gradcam_engine.generate_heatmap(input_tensor, target_class=target_class)
        _, gcam_overlay = apply_colormap_on_image(rgb_disp, gcam_map, alpha=0.52)
        gcam_p = output_subdir / "gradcam_overlay.png"
        Image.fromarray(gcam_overlay).save(gcam_p)

        # 2. LayerCAM
        lcam_map, _, _ = layercam_engine.generate_heatmap(input_tensor, target_class=target_class)
        _, lcam_overlay = apply_colormap_on_image(rgb_disp, lcam_map, alpha=0.52)
        lcam_p = output_subdir / "layercam_overlay.png"
        Image.fromarray(lcam_overlay).save(lcam_p)

        # 3. Guided Backprop
        _, gb_gray, _, _ = guided_engine.generate_saliency(input_tensor, target_class=target_class)
        gb_p = output_subdir / "guided_backprop.png"
        Image.fromarray(np.uint8(gb_gray * 255)).save(gb_p)

        # 4. Guided Grad-CAM
        _, ggcam_gray = guided_engine.generate_guided_gradcam(input_tensor, gcam_map, target_class=target_class)
        ggcam_p = output_subdir / "guided_gradcam.png"
        Image.fromarray(np.uint8(ggcam_gray * 255)).save(ggcam_p)

    finally:
        gradcam_engine.remove_hooks()
        layercam_engine.remove_hooks()
        guided_engine.remove_hooks()

    return {
        "gradcam_overlay": f"generated/{output_subdir.name}/gradcam_overlay.png",
        "layercam_overlay": f"generated/{output_subdir.name}/layercam_overlay.png",
        "guided_backprop": f"generated/{output_subdir.name}/guided_backprop.png",
        "guided_gradcam": f"generated/{output_subdir.name}/guided_gradcam.png"
    }


def explain_classical_ml(model, model_id: str, features_vec: np.ndarray, output_subdir: Path, pred_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates SHAP explanation for the single prediction.
    """
    output_subdir.mkdir(parents=True, exist_ok=True)
    import shap
    feature_names = get_feature_names()
    X_single = features_vec.reshape(1, -1)

    top_pos_features = []
    top_neg_features = []
    waterfall_rel_path = None

    try:
        if model_id in ["random_forest", "decision_tree"]:
            tree_model = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
            scaler = model.named_steps["scaler"] if hasattr(model, "named_steps") else None
            X_scaled = scaler.transform(X_single) if scaler else X_single
            explainer = shap.TreeExplainer(tree_model)
            sv = explainer.shap_values(X_scaled)
            if isinstance(sv, list):
                shap_vals = sv[1][0] if len(sv) > 1 else sv[0][0]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            elif sv.ndim == 3:
                shap_vals = sv[0, :, 1]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                shap_vals = sv[0]
                base_val = explainer.expected_value

        elif model_id == "logistic_regression":
            lr_model = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
            scaler = model.named_steps["scaler"] if hasattr(model, "named_steps") else None
            X_scaled = scaler.transform(X_single) if scaler else X_single
            X_bg = get_shap_background()
            X_bg_scaled = scaler.transform(X_bg) if scaler else X_bg
            explainer = shap.LinearExplainer(lr_model, X_bg_scaled)
            sv = explainer.shap_values(X_scaled)
            shap_vals = sv[0]
            base_val = explainer.expected_value

        elif model_id in ["svm", "knn"]:
            # Bounded KernelExplainer
            X_bg = get_shap_background()
            def predict_fn(x):
                return model.predict_proba(x)[:, 1]
            explainer = shap.KernelExplainer(predict_fn, X_bg)
            sv = explainer.shap_values(X_single, nsamples=100)
            shap_vals = sv[0]
            base_val = explainer.expected_value

        else:
            return {"available": False, "reason": "SHAP unsupported for this model"}

        # Extract top 3 positive and top 3 negative features
        sorted_pos = np.argsort(shap_vals)[::-1]
        for idx in sorted_pos[:3]:
            if shap_vals[idx] > 0:
                top_pos_features.append({
                    "name": feature_names[idx],
                    "value": round(float(X_single[0, idx]), 4),
                    "shap": round(float(shap_vals[idx]), 4)
                })

        sorted_neg = np.argsort(shap_vals)
        for idx in sorted_neg[:3]:
            if shap_vals[idx] < 0:
                top_neg_features.append({
                    "name": feature_names[idx],
                    "value": round(float(X_single[0, idx]), 4),
                    "shap": round(float(shap_vals[idx]), 4)
                })

        # Save Waterfall / Bar plot
        exp_obj = shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=X_single[0],
            feature_names=feature_names
        )
        plt.figure(figsize=(9, 4.8))
        shap.plots.waterfall(exp_obj, max_display=10, show=False)
        plt.title(f"SHAP Attribution: {model_id.upper()}\nPred: {pred_info['predicted_class']} (Prob Anemia: {pred_info.get('prob_anemia', 'N/A')})", fontsize=10, fontweight="bold")
        plt.tight_layout()
        wf_path = output_subdir / "shap_waterfall.png"
        plt.savefig(wf_path, dpi=180, bbox_inches="tight")
        plt.close()
        waterfall_rel_path = f"generated/{output_subdir.name}/shap_waterfall.png"

        return {
            "available": True,
            "waterfall_plot": waterfall_rel_path,
            "top_positive": top_pos_features,
            "top_negative": top_neg_features
        }

    except Exception as e:
        print(f"Error computing SHAP: {e}")
        return {"available": False, "reason": str(e)}
