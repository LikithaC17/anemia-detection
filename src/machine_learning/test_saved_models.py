"""
Validation & Inference Test for Saved Classical ML Models
Verifies that all serialized joblib models can be successfully loaded and execute inference.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
MODELS_DIR = WORKSPACE_ROOT / "models" / "classical_ml"
VAL_CSV = WORKSPACE_ROOT / "results" / "phase3" / "features" / "validation_features.csv"


def test_saved_models():
    model_files = sorted(list(MODELS_DIR.glob("*.joblib")))
    print(f"Discovered {len(model_files)} serialized model files in {MODELS_DIR}:")

    val_df = pd.read_csv(VAL_CSV)
    meta_cols = [
        "image_path", "relative_path", "dataset", "modality", "anatomy_subcategory",
        "subject_id", "cluster_id", "class_label", "target_binary",
        "sha256", "width", "height", "channels", "format"
    ]
    feat_cols = [c for c in val_df.columns if c not in meta_cols]
    sample_X = val_df[feat_cols].iloc[:5].values.astype(np.float32)

    passed = 0
    failed = 0

    for mf in model_files:
        try:
            model = joblib.load(mf)
            preds = model.predict(sample_X)
            probs = model.predict_proba(sample_X)[:, 1] if hasattr(model, "predict_proba") else None
            prob_str = f"Prob: {min(probs):.3f} - {max(probs):.3f}" if probs is not None else "Prob: N/A"
            print(f"  [PASS] {mf.name:<38} | Preds: {preds} | {prob_str}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {mf.name}: {e}")
            failed += 1

    print(f"\nModel Serialization Audit: {passed} passed, {failed} failed.")
    return failed == 0


if __name__ == "__main__":
    success = test_saved_models()
    exit(0 if success else 1)
