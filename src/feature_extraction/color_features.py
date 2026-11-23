"""
Color Feature Extraction Module for Anemia Detection
Extracts RGB and HSV histograms, channel statistics, and clinical erythema/pallor indices.
"""

import numpy as np
import cv2
from PIL import Image


def extract_color_features(img_rgb: np.ndarray, bins_per_channel: int = 16) -> tuple[np.ndarray, list[str]]:
    """
    Extracts comprehensive colorimetric features relevant to pallor detection.
    
    Parameters:
        img_rgb: numpy array of shape (H, W, 3) in uint8 RGB format.
        bins_per_channel: Number of histogram bins per channel.
        
    Returns:
        feature_vector: 1D numpy array of float32 values.
        feature_names: List of strings naming each feature.
    """
    features = []
    names = []

    # 1. RGB Mean and Standard Deviation
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    features.extend([
        float(np.mean(r)), float(np.mean(g)), float(np.mean(b)),
        float(np.std(r)), float(np.std(g)), float(np.std(b))
    ])
    names.extend([
        "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
        "rgb_std_r", "rgb_std_g", "rgb_std_b"
    ])

    # 2. Clinical Pallor and Erythema Ratios
    # In anemia screening, pallor manifests as reduced redness relative to other bands.
    eps = 1e-6
    total_intensity = r + g + b + eps
    norm_r = r / total_intensity
    norm_g = g / total_intensity
    norm_b = b / total_intensity

    # Erythema index: log(R) - log(G)
    erythema_index = np.log(r + 1.0) - np.log(g + 1.0)
    # Red-Green difference ratio (Relative Pallor Index)
    rg_ratio = (r - g) / (r + g + eps)

    features.extend([
        float(np.mean(norm_r)), float(np.mean(norm_g)), float(np.mean(norm_b)),
        float(np.mean(erythema_index)), float(np.std(erythema_index)),
        float(np.mean(rg_ratio)), float(np.std(rg_ratio))
    ])
    names.extend([
        "norm_r_mean", "norm_g_mean", "norm_b_mean",
        "erythema_index_mean", "erythema_index_std",
        "rg_ratio_mean", "rg_ratio_std"
    ])

    # 3. HSV Color Space Statistics
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h = img_hsv[:, :, 0].astype(np.float32)
    s = img_hsv[:, :, 1].astype(np.float32)
    v = img_hsv[:, :, 2].astype(np.float32)

    features.extend([
        float(np.mean(h)), float(np.mean(s)), float(np.mean(v)),
        float(np.std(h)), float(np.std(s)), float(np.std(v))
    ])
    names.extend([
        "hsv_mean_h", "hsv_mean_s", "hsv_mean_v",
        "hsv_std_h", "hsv_std_s", "hsv_std_v"
    ])

    # 4. RGB Histograms (normalized)
    for c_idx, c_name in enumerate(["r", "g", "b"]):
        hist, _ = np.histogram(img_rgb[:, :, c_idx], bins=bins_per_channel, range=(0, 256), density=True)
        features.extend(hist.astype(float).tolist())
        names.extend([f"rgb_hist_{c_name}_bin{b_i}" for b_i in range(bins_per_channel)])

    # 5. HSV Histograms (normalized)
    # Hue ranges 0-180 in OpenCV; Saturation and Value range 0-256
    hist_h, _ = np.histogram(h, bins=bins_per_channel, range=(0, 180), density=True)
    hist_s, _ = np.histogram(s, bins=bins_per_channel, range=(0, 256), density=True)
    hist_v, _ = np.histogram(v, bins=bins_per_channel, range=(0, 256), density=True)

    features.extend(hist_h.astype(float).tolist())
    names.extend([f"hsv_hist_h_bin{b_i}" for b_i in range(bins_per_channel)])
    features.extend(hist_s.astype(float).tolist())
    names.extend([f"hsv_hist_s_bin{b_i}" for b_i in range(bins_per_channel)])
    features.extend(hist_v.astype(float).tolist())
    names.extend([f"hsv_hist_v_bin{b_i}" for b_i in range(bins_per_channel)])

    return np.array(features, dtype=np.float32), names
