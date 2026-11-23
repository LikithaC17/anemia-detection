"""
Support Vector Machine (SVM) Training Module
Pipeline: StandardScaler -> SVC (with probability=True, class_weight='balanced')
Tuning: Regularization C, kernel, gamma
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
MODELS_DIR = WORKSPACE_ROOT / "models" / "classical_ml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              random_state: int = 42, fast_tune: bool = False) -> tuple[Pipeline, dict]:
    """
    Trains and tunes an SVM classifier with balanced class weighting.
    """
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, class_weight="balanced", random_state=random_state))
    ])

    param_grid = {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__kernel": ["rbf"],
        "svm__gamma": ["scale", 0.01]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        base_pipeline,
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
        "model_name": "Support Vector Machine (SVM)",
        "best_params": best_params,
        "best_cv_roc_auc": float(grid_search.best_score_)
    }


def save_svm_model(model: Pipeline, filename: str = "svm_model.joblib") -> Path:
    out_path = MODELS_DIR / filename
    joblib.dump(model, out_path)
    return out_path
