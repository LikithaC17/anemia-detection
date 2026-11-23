"""
Random Forest Classifier Training Module
Pipeline: RandomForestClassifier (with class_weight='balanced')
Tuning: n_estimators, max_depth, min_samples_split, max_features
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
MODELS_DIR = WORKSPACE_ROOT / "models" / "classical_ml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                         random_state: int = 42, fast_tune: bool = False) -> tuple[RandomForestClassifier, dict]:
    """
    Trains and tunes a Random Forest classifier with balanced class weighting.
    """
    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=4,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, {
        "model_name": "Random Forest",
        "best_params": best_params,
        "best_cv_roc_auc": float(grid_search.best_score_)
    }


def save_rf_model(model: RandomForestClassifier, filename: str = "random_forest_model.joblib") -> Path:
    out_path = MODELS_DIR / filename
    joblib.dump(model, out_path)
    return out_path
