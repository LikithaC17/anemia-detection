"""
Flask Application Configuration
Project: AI-Based Anemia Detection Using Image Processing, ML, DL, and XAI
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "anemia-detection-academic-research-prototype-key-2026")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    GENERATED_FOLDER = BASE_DIR / "static" / "generated"
    
    MODELS_DIR = BASE_DIR / "models"
    CLASSICAL_ML_DIR = MODELS_DIR / "classical_ml"
    DEEP_LEARNING_DIR = MODELS_DIR / "deep_learning"
    
    RESULTS_DIR = BASE_DIR / "results"
    PHASE2_RESULTS = RESULTS_DIR / "phase2"
    PHASE3_RESULTS = RESULTS_DIR / "phase3"
    PHASE4_RESULTS = RESULTS_DIR / "phase4"
    PHASE5_RESULTS = RESULTS_DIR / "phase5"
    PHASE6_RESULTS = RESULTS_DIR / "phase6"
    PHASE7_RESULTS = RESULTS_DIR / "phase7"
    
    # Supported modalities and their models based on verified Phases 1-6
    SUPPORTED_MODALITIES = {
        "eye_conjunctiva": {
            "display_name": "Eye Conjunctiva",
            "models": {
                "resnet18": {"name": "Pretrained ResNet-18", "type": "Deep Learning", "supports_cam": True},
                "custom_cnn": {"name": "Custom CNN", "type": "Deep Learning", "supports_cam": True},
                "svm": {"name": "Support Vector Machine (RBF)", "type": "Classical ML", "supports_shap": True},
                "random_forest": {"name": "Random Forest", "type": "Classical ML", "supports_shap": True},
                "logistic_regression": {"name": "Logistic Regression", "type": "Classical ML", "supports_shap": True},
                "knn": {"name": "K-Nearest Neighbors", "type": "Classical ML", "supports_shap": True},
                "decision_tree": {"name": "Decision Tree", "type": "Classical ML", "supports_shap": True}
            }
        },
        "palmar_pallor": {
            "display_name": "Palmar Pallor",
            "models": {
                "svm": {"name": "Support Vector Machine (RBF)", "type": "Classical ML", "supports_shap": False},
                "random_forest": {"name": "Random Forest", "type": "Classical ML", "supports_shap": False},
                "logistic_regression": {"name": "Logistic Regression", "type": "Classical ML", "supports_shap": False},
                "knn": {"name": "K-Nearest Neighbors", "type": "Classical ML", "supports_shap": False},
                "decision_tree": {"name": "Decision Tree", "type": "Classical ML", "supports_shap": False}
            }
        },
        "nail_beds": {
            "display_name": "Nail Beds",
            "models": {
                "svm": {"name": "Support Vector Machine (RBF)", "type": "Classical ML", "supports_shap": False},
                "random_forest": {"name": "Random Forest", "type": "Classical ML", "supports_shap": False},
                "logistic_regression": {"name": "Logistic Regression", "type": "Classical ML", "supports_shap": False},
                "knn": {"name": "K-Nearest Neighbors", "type": "Classical ML", "supports_shap": False},
                "decision_tree": {"name": "Decision Tree", "type": "Classical ML", "supports_shap": False}
            }
        }
    }
