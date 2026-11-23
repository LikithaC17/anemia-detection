"""
Feature Importance Analysis Module
Extracts Random Forest Gini importance and Permutation Importance on validation set.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
FI_DIR = WORKSPACE_ROOT / "results" / "phase3" / "feature_importance"
FI_DIR.mkdir(parents=True, exist_ok=True)


def compute_and_plot_feature_importance(model, X_val: np.ndarray, y_val: np.ndarray,
                                        feature_names: list[str], modality_name: str,
                                        top_n: int = 20) -> pd.DataFrame:
    """
    Computes both Gini impurity-based and Permutation-based feature importances on validation data.
    """
    # 1. Random Forest Gini Importance
    rf_estimator = model.named_steps["rf"] if hasattr(model, "named_steps") and "rf" in model.named_steps else model
    if hasattr(rf_estimator, "feature_importances_"):
        gini_importances = rf_estimator.feature_importances_
    else:
        gini_importances = np.zeros(len(feature_names))

    # 2. Permutation Importance on Validation Data (Scored by ROC-AUC)
    print(f"Computing permutation importance for {modality_name} on validation set...", flush=True)
    perm_result = permutation_importance(
        model, X_val, y_val,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=4
    )
    perm_mean = perm_result.importances_mean
    perm_std = perm_result.importances_std

    # Compile DataFrame
    fi_df = pd.DataFrame({
        "feature_name": feature_names,
        "gini_importance": gini_importances,
        "permutation_importance_mean": perm_mean,
        "permutation_importance_std": perm_std
    })

    # Sort by permutation importance
    fi_df = fi_df.sort_values(by="permutation_importance_mean", ascending=False).reset_index(drop=True)

    # Save CSV
    safe_mod = modality_name.lower().replace(" ", "_")
    csv_path = FI_DIR / f"feature_importance_{safe_mod}.csv"
    fi_df.to_csv(csv_path, index=False)
    print(f"Feature importance table saved to: {csv_path}")

    # Plot 1: Top N Gini Importance
    top_gini = fi_df.sort_values(by="gini_importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, 7))
    plt.barh(top_gini["feature_name"][::-1], top_gini["gini_importance"][::-1], color="#2980b9", edgecolor="black")
    plt.xlabel("Gini Impurity Reduction (MDI)", fontsize=11)
    plt.title(f"Top {top_n} Features by Random Forest Gini Importance ({modality_name})", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    gini_chart_path = FI_DIR / f"feature_importance_rf_{safe_mod}.png"
    plt.savefig(gini_chart_path, dpi=200)
    plt.close()

    # Plot 2: Top N Permutation Importance
    top_perm = fi_df.head(top_n)
    plt.figure(figsize=(10, 7))
    plt.barh(top_perm["feature_name"][::-1], top_perm["permutation_importance_mean"][::-1],
             xerr=top_perm["permutation_importance_std"][::-1], color="#e74c3c", edgecolor="black", capsize=3)
    plt.xlabel("Mean ROC-AUC Drop Upon Feature Permutation", fontsize=11)
    plt.title(f"Top {top_n} Features by Permutation Importance ({modality_name})\n(Evaluated on Hold-out Validation Set)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    perm_chart_path = FI_DIR / f"permutation_importance_{safe_mod}.png"
    plt.savefig(perm_chart_path, dpi=200)
    plt.close()

    return fi_df
