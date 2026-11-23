"""
Utility functions for Explainable AI (XAI)
Provides tensor denormalization, colormap overlay, and representative case selection.

Author: Senior Machine Learning Engineer & Explainable AI Specialist
Project: AI-Based Anemia Detection
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2


def denormalize_tensor(tensor: torch.Tensor,
                       mean: List[float] = [0.485, 0.456, 0.406],
                       std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Converts a normalized PyTorch image tensor (C, H, W) back to uint8 RGB numpy array (H, W, C).
    """
    img_t = tensor.clone().detach().cpu()
    if img_t.ndim == 4:
        img_t = img_t.squeeze(0)

    for c in range(3):
        img_t[c] = img_t[c] * std[c] + mean[c]

    img_np = img_t.numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np


def apply_colormap_on_image(rgb_img: np.ndarray,
                           heatmap: np.ndarray,
                           alpha: float = 0.55,
                           colormap: int = cv2.COLORMAP_JET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Overlays a 2D heatmap [0, 1] onto an RGB image.
    Returns: (colored_heatmap_rgb, blended_overlay_rgb)
    """
    h, w, _ = rgb_img.shape
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))

    # Apply colormap in BGR then convert to RGB
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original RGB
    overlay = cv2.addWeighted(colored_heatmap_rgb, alpha, rgb_img, 1.0 - alpha, 0)
    return colored_heatmap_rgb, overlay


def select_representative_samples(df_manifest: pd.DataFrame,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_prob: np.ndarray,
                                  n_per_category: int = 2) -> List[Dict[str, Any]]:
    """
    Selects representative individual test samples across all diagnostic outcomes:
      - True Positive (High Confidence Anemia detection)
      - True Negative (High Confidence Non_Anemia detection)
      - False Negative (Missed Anemia / clinical error)
      - False Positive (False Alarm)
      - Borderline / Uncertain (Probability closest to 0.50)
    """
    df = df_manifest.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["prob_anemia"] = y_prob
    df["prob_margin"] = (df["prob_anemia"] - 0.5).abs()

    selected = []

    # 1. True Positives (High Confidence: highest prob_anemia where true=1, pred=1)
    tp_df = df[(df["y_true"] == 1) & (df["y_pred"] == 1)].sort_values(by="prob_anemia", ascending=False)
    for idx in range(min(n_per_category, len(tp_df))):
        row = tp_df.iloc[idx]
        selected.append({
            "category": "True Positive (High Confidence)",
            "index": int(row.name),
            "image_path": str(row["image_path"]),
            "subject_id": str(row.get("subject_id", "Unknown")),
            "true_label": "Anemia",
            "predicted_label": "Anemia",
            "confidence": float(row["prob_anemia"]),
            "notes": "Correctly identified anemic pallor with high model confidence"
        })

    # 2. True Negatives (High Confidence: lowest prob_anemia where true=0, pred=0)
    tn_df = df[(df["y_true"] == 0) & (df["y_pred"] == 0)].sort_values(by="prob_anemia", ascending=True)
    for idx in range(min(n_per_category, len(tn_df))):
        row = tn_df.iloc[idx]
        selected.append({
            "category": "True Negative (High Confidence)",
            "index": int(row.name),
            "image_path": str(row["image_path"]),
            "subject_id": str(row.get("subject_id", "Unknown")),
            "true_label": "Non_Anemia",
            "predicted_label": "Non_Anemia",
            "confidence": float(1.0 - row["prob_anemia"]),
            "notes": "Correctly identified healthy conjunctival vascularity"
        })

    # 3. False Negatives (Critical Miss: true=1, pred=0)
    fn_df = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].sort_values(by="prob_anemia", ascending=True)
    for idx in range(min(n_per_category, len(fn_df))):
        row = fn_df.iloc[idx]
        selected.append({
            "category": "False Negative (Missed Case)",
            "index": int(row.name),
            "image_path": str(row["image_path"]),
            "subject_id": str(row.get("subject_id", "Unknown")),
            "true_label": "Anemia",
            "predicted_label": "Non_Anemia",
            "confidence": float(1.0 - row["prob_anemia"]),
            "notes": "Anemic subject misclassified as healthy (false negative error)"
        })

    # 4. False Positives (False Alarm: true=0, pred=1)
    fp_df = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].sort_values(by="prob_anemia", ascending=False)
    for idx in range(min(n_per_category, len(fp_df))):
        row = fp_df.iloc[idx]
        selected.append({
            "category": "False Positive (False Alarm)",
            "index": int(row.name),
            "image_path": str(row["image_path"]),
            "subject_id": str(row.get("subject_id", "Unknown")),
            "true_label": "Non_Anemia",
            "predicted_label": "Anemia",
            "confidence": float(row["prob_anemia"]),
            "notes": "Healthy subject misclassified as anemic (false positive error)"
        })

    # 5. Borderline / Low-Confidence Cases (prob closest to 0.50)
    border_df = df.sort_values(by="prob_margin", ascending=True)
    for idx in range(min(n_per_category, len(border_df))):
        row = border_df.iloc[idx]
        t_lbl = "Anemia" if row["y_true"] == 1 else "Non_Anemia"
        p_lbl = "Anemia" if row["y_pred"] == 1 else "Non_Anemia"
        conf = float(row["prob_anemia"] if p_lbl == "Anemia" else 1.0 - row["prob_anemia"])
        selected.append({
            "category": "Borderline / Low Confidence",
            "index": int(row.name),
            "image_path": str(row["image_path"]),
            "subject_id": str(row.get("subject_id", "Unknown")),
            "true_label": t_lbl,
            "predicted_label": p_lbl,
            "confidence": conf,
            "notes": f"High diagnostic uncertainty (P(Anemia)={row['prob_anemia']:.3f})"
        })

    return selected
