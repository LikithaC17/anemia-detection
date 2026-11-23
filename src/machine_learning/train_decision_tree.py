"""
Decision Tree Classifier Training Module
Pipeline: DecisionTreeClassifier (with class_weight='balanced')
Tuning: max_depth, min_samples_split, criterion
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
MODELS_DIR = WORKSPACE_ROOT / "models" / "classical_ml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                        random_state: int = 42, fast_tune: bool = False) -> tuple[DecisionTreeClassifier, dict]:
    """
    Trains and tunes a Decision Tree classifier with balanced class weighting.
    """
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        dt,
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
        "model_name": "Decision Tree",
        "best_params": best_params,
        "best_cv_roc_auc": float(grid_search.best_score_)
    }


def save_dt_model(model: DecisionTreeClassifier, filename: str = "decision_tree_model.joblib") -> Path:
    out_path = MODELS_DIR / filename
    joblib.dump(model, out_path)
    return out_path
