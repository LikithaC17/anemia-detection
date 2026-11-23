"""
Comprehensive Feature Extraction Pipeline for Classical ML
Extracts color and texture features across Train, Validation, and Test splits.
Author: Senior Machine Learning Engineer & Python Developer
"""

import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Ensure project root is in sys.path
WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.feature_extraction.color_features import extract_color_features
from src.feature_extraction.texture_features import extract_texture_features

RESULTS_PHASE2 = WORKSPACE_ROOT / "results" / "phase2"
FEATURES_DIR = WORKSPACE_ROOT / "results" / "phase3" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def extract_features_single_image(row_tuple: tuple) -> tuple:
    """
    Worker function to extract features for a single image row.
    Returns: (idx, success, feature_dict_or_error)
    """
    idx, row_dict = row_tuple
    img_path = row_dict["image_path"]

    try:
        # Load image via PIL to verify and convert
        with Image.open(img_path) as pil_img:
            img_rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)

        # Extract features
        color_feats, color_names = extract_color_features(img_rgb)
        texture_feats, texture_names = extract_texture_features(img_rgb)

        all_feats = np.concatenate([color_feats, texture_feats])
        all_names = color_names + texture_names

        feat_dict = {name: float(val) for name, val in zip(all_names, all_feats)}
        return (idx, True, feat_dict)

    except Exception as e:
        return (idx, False, str(e))


def process_split_dataframe(df: pd.DataFrame, split_name: str, max_workers: int = 8) -> pd.DataFrame:
    """
    Extracts features for all rows in a split DataFrame in parallel.
    """
    print(f"\nProcessing {split_name} split ({len(df)} images)...")
    tasks = [(idx, row.to_dict()) for idx, row in df.iterrows()]

    extracted_records = [None] * len(df)
    errors = []

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_features_single_image, task): i for i, task in enumerate(tasks)}
        completed = 0
        total = len(tasks)

        for future in as_completed(futures):
            i = futures[future]
            completed += 1
            idx, success, result = future.result()
            if success:
                extracted_records[i] = result
            else:
                errors.append((idx, result))

            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - start_time
                fps = completed / max(elapsed, 0.001)
                print(f"  [{split_name}] Completed {completed}/{total} images ({fps:.1f} img/s)")

    if errors:
        print(f"  [WARNING] {len(errors)} images failed extraction in {split_name} split!")
        for err in errors[:5]:
            print(f"    - Index {err[0]}: {err[1]}")

    # Combine metadata with extracted features
    feat_df = pd.DataFrame(extracted_records)
    combined_df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return combined_df


def main():
    print("=" * 80)
    print("PHASE 3: PARALLEL FEATURE EXTRACTION PIPELINE")
    print("=" * 80)

    # 1. Verification of inputs from Phase 2
    for fname in ["train.csv", "validation.csv", "test.csv", "verified_labels.csv", "split_summary.json"]:
        fpath = RESULTS_PHASE2 / fname
        if not fpath.exists():
            print(f"[FATAL ERROR] Required Phase 2 output not found: {fpath}")
            sys.exit(1)

    train_df = pd.read_csv(RESULTS_PHASE2 / "train.csv")
    val_df = pd.read_csv(RESULTS_PHASE2 / "validation.csv")
    test_df = pd.read_csv(RESULTS_PHASE2 / "test.csv")

    print(f"Loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Get sample feature names
    sample_img = np.zeros((224, 224, 3), dtype=np.uint8)
    _, c_names = extract_color_features(sample_img)
    _, t_names = extract_texture_features(sample_img)
    all_feature_names = c_names + t_names

    with open(FEATURES_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_features": len(all_feature_names),
            "color_feature_count": len(c_names),
            "texture_feature_count": len(t_names),
            "feature_names": all_feature_names
        }, f, indent=2)
    print(f"Feature schema saved: {len(all_feature_names)} features ({len(c_names)} color, {len(t_names)} texture)")

    # 2. Extract features for each split
    workers = min(os.cpu_count() or 4, 10)
    print(f"Using {workers} parallel CPU workers.")

    train_feats = process_split_dataframe(train_df, "Train", max_workers=workers)
    val_feats = process_split_dataframe(val_df, "Validation", max_workers=workers)
    test_feats = process_split_dataframe(test_df, "Test", max_workers=workers)

    # 3. Save master feature CSVs
    print("\nSaving feature matrices...")
    train_feats.to_csv(FEATURES_DIR / "train_features.csv", index=False)
    val_feats.to_csv(FEATURES_DIR / "validation_features.csv", index=False)
    test_feats.to_csv(FEATURES_DIR / "test_features.csv", index=False)
    print("  Master feature CSVs saved successfully.")

    # 4. Save per-modality slices for clean, modular modeling
    for mod in ["Eye", "Nail Beds", "Palm"]:
        mod_clean = mod.lower().replace(" ", "_")
        m_train = train_feats[train_feats["modality"] == mod]
        m_val = val_feats[val_feats["modality"] == mod]
        m_test = test_feats[test_feats["modality"] == mod]

        m_train.to_csv(FEATURES_DIR / f"{mod_clean}_train_features.csv", index=False)
        m_val.to_csv(FEATURES_DIR / f"{mod_clean}_val_features.csv", index=False)
        m_test.to_csv(FEATURES_DIR / f"{mod_clean}_test_features.csv", index=False)
        print(f"  Saved {mod} slice: train={len(m_train)}, val={len(m_val)}, test={len(m_test)}")

    print("\nFEATURE EXTRACTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
