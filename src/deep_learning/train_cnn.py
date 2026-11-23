"""
Training Script for Custom CNN on Anemia Detection
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.deep_learning.data_loader import create_dataloaders
from src.deep_learning.custom_cnn import get_custom_cnn
from src.deep_learning.train_model import train_model
from src.deep_learning.evaluate_dl import evaluate_deep_learning_model

MODELS_DIR = WORKSPACE_ROOT / "models" / "deep_learning" / "cnn"
HISTORY_DIR = WORKSPACE_ROOT / "results" / "phase4" / "training_history"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def run_training_custom_cnn(
    modality: str = "Eye",
    batch_size: int = 32,
    lr: float = 3e-4,
    num_epochs: int = 10,
    patience: int = 4
) -> dict:
    """
    Trains and evaluates Custom CNN for a specified modality.
    """
    train_csv = WORKSPACE_ROOT / "results" / "phase2" / "train.csv"
    val_csv = WORKSPACE_ROOT / "results" / "phase2" / "validation.csv"
    mod_clean = modality.lower().replace(" ", "_")

    print("=" * 80, flush=True)
    print(f"TRAINING CUSTOM CNN ON MODALITY: {modality.upper()}", flush=True)
    print("=" * 80, flush=True)

    train_loader, val_loader, _, train_ds = create_dataloaders(
        train_manifest=train_csv,
        val_manifest=val_csv,
        modality=modality,
        batch_size=batch_size,
        num_workers=0
    )

    class_weights = train_ds.get_class_weights()
    print(f"Modality: {modality} | Train Images: {len(train_ds)} | Val Images: {len(val_loader.dataset)}", flush=True)
    print(f"Class Weights (Cross-Entropy): {class_weights.tolist()}", flush=True)

    model = get_custom_cnn(num_classes=2, dropout_rate=0.4)
    chk_path = MODELS_DIR / f"best_custom_cnn_{mod_clean}.pth"

    trained_model, history, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        lr=lr,
        weight_decay=1e-4,
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_path=chk_path,
        model_name="Custom CNN",
        modality_name=modality
    )

    # Save history JSON
    history_path = HISTORY_DIR / f"history_custom_cnn_{mod_clean}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Comprehensive evaluation
    eval_metrics = evaluate_deep_learning_model(
        y_true=best_metrics["y_true"],
        y_pred=best_metrics["y_pred"],
        y_prob=best_metrics["y_prob"],
        model_name="Custom CNN",
        modality_name=modality,
        history=history,
        training_time=best_metrics.get("total_training_time_seconds", 0.0)
    )

    print("\nCustom CNN Validation Results:", flush=True)
    for k, v in eval_metrics.items():
        if k != "curve_data":
            print(f"  {k}: {v}", flush=True)

    return eval_metrics


if __name__ == "__main__":
    mod = sys.argv[1] if len(sys.argv) > 1 else "Eye"
    run_training_custom_cnn(modality=mod)
