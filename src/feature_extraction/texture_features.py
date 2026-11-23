"""
Texture Feature Extraction Module for Anemia Detection
Extracts Gray-Level Co-occurrence Matrix (GLCM) statistics and Local Binary Patterns (LBP).
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_texture_features(img_gray: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extracts second-order spatial gray-level co-occurrence statistics and LBP texture descriptor.
    
    Parameters:
        img_gray: 2D numpy array uint8 grayscale image.
        
    Returns:
        feature_vector: 1D numpy array of float32 values.
        feature_names: List of strings naming each feature.
    """
    features = []
    names = []

    # Ensure 2D uint8
    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    if img_gray.dtype != np.uint8:
        img_gray = (np.clip(img_gray, 0, 255)).astype(np.uint8)

    # Standardize scale to 256x256 for uniform spatial frequency and fast feature computation
    if img_gray.shape[0] > 256 or img_gray.shape[1] > 256:
        img_gray = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA)

    # 1. Gray-Level Co-occurrence Matrix (GLCM)
    # Quantize to 64 levels for robust co-occurrence density estimation and high computational speed
    quantized_gray = (img_gray // 4).astype(np.uint8)
    distances = [1, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        quantized_gray,
        distances=distances,
        angles=angles,
        levels=64,
        symmetric=True,
        normed=True
    )

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for prop in properties:
        val_matrix = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
        for d_idx, dist in enumerate(distances):
            # Mean and std across the 4 directional angles
            mean_val = float(np.mean(val_matrix[d_idx, :]))
            std_val = float(np.std(val_matrix[d_idx, :]))
            features.extend([mean_val, std_val])
            names.extend([
                f"glcm_{prop}_d{dist}_mean",
                f"glcm_{prop}_d{dist}_std"
            ])

    # 2. Local Binary Patterns (LBP)
    # P=8 neighbours, R=1 radius with rotation-invariant uniform patterns
    radius = 1
    n_points = 8
    lbp = local_binary_pattern(img_gray, P=n_points, R=radius, method="uniform")
    n_bins = n_points + 2  # 10 bins for uniform LBP
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    features.extend(lbp_hist.astype(float).tolist())
    names.extend([f"lbp_bin_{b_i}" for b_i in range(n_bins)])

    # 3. Spatial Gradient Magnitude Statistics (Sobel Edge Energy)
    sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)

    features.extend([
        float(np.mean(grad_mag)),
        float(np.std(grad_mag))
    ])
    names.extend([
        "grad_magnitude_mean",
        "grad_magnitude_std"
    ])

    return np.array(features, dtype=np.float32), names
