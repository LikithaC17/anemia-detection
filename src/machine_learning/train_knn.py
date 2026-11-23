"""
K-Nearest Neighbors (KNN) Training Module
Pipeline: StandardScaler -> KNeighborsClassifier
Tuning: n_neighbors, weights
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
MODELS_DIR = WORKSPACE_ROOT / "models" / "classical_ml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_knn(X_train: np.ndarray, y_train: np.ndarray,
              random_state: int = 42, fast_tune: bool = False) -> tuple[Pipeline, dict]:
    """
    Trains and tunes a K-Nearest Neighbors classifier with feature scaling.
    """
    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_jobs=1))
    ])

    param_grid = {
        "knn__n_neighbors": [3, 5, 9, 15],
        "knn__weights": ["uniform", "distance"]
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
        "model_name": "K-Nearest Neighbors (KNN)",
        "best_params": best_params,
        "best_cv_roc_auc": float(grid_search.best_score_)
    }


def save_knn_model(model: Pipeline, filename: str = "knn_model.joblib") -> Path:
    out_path = MODELS_DIR / filename
    joblib.dump(model, out_path)
    return out_path
