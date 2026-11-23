"""
Automated Test Suite for Phase 7: Flask Web Application
Validates application startup, page routing, secure uploads,
Classical ML and Deep Learning inference, real-time XAI generation,
error handling, and disclaimer presentation.

Author: Senior Machine Learning & QA Engineer
Project: AI-Based Anemia Detection Using Image Processing, ML, DL, and XAI
"""

import io
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from app import app
from config import Config
from src.inference_service import (
    load_model,
    extract_image_features,
    predict_classical_ml,
    predict_deep_learning,
    explain_deep_learning,
    explain_classical_ml
)

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


@pytest.fixture
def real_eye_image_bytes():
    df = pd.read_csv(WORKSPACE_ROOT / "results" / "phase2" / "test.csv")
    eye_path = df[df["modality"] == "Eye"].iloc[0]["image_path"]
    with open(eye_path, "rb") as f:
        return f.read(), Path(eye_path).name


@pytest.fixture
def real_palm_image_bytes():
    df = pd.read_csv(WORKSPACE_ROOT / "results" / "phase2" / "test.csv")
    palm_path = df[df["modality"] == "Palm"].iloc[0]["image_path"]
    with open(palm_path, "rb") as f:
        return f.read(), Path(palm_path).name


def test_flask_app_startup_and_config():
    assert app is not None
    assert app.config["SECRET_KEY"] is not None
    assert app.config["MAX_CONTENT_LENGTH"] == 16 * 1024 * 1024
    assert "eye_conjunctiva" in Config.SUPPORTED_MODALITIES
    assert "palmar_pallor" in Config.SUPPORTED_MODALITIES
    assert "nail_beds" in Config.SUPPORTED_MODALITIES


def test_home_page_response(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "AnemiaScan AI" in html
    assert "Eye Conjunctiva" in html
    assert "Palmar Pallor" in html
    assert "Nail Beds" in html
    assert "Academic Research Prototype" in html


def test_prediction_page_response(client):
    response = client.get("/prediction")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Image Analysis & Anemia Prediction" in html
    assert "upload-dropzone" in html
    assert "modality-select" in html
    assert "model-select" in html


def test_dashboard_page_response(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Model Evaluation Dashboard" in html
    assert "17" in html  # 17 models evaluated
    assert "master_performance_comparison.png" in html
    assert "training_curves.png" in html


def test_model_comparison_page_response(client):
    response = client.get("/model_comparison")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Model Architecture & Modality Comparison" in html
    assert "Pretrained ResNet-18" in html
    assert "Custom 4-Block CNN" in html


def test_explainable_ai_page_response(client):
    response = client.get("/explainable_ai")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Explainable Artificial Intelligence" in html
    assert "LayerCAM" in html
    assert "Guided Backpropagation" in html
    assert "SHAP" in html


def test_about_page_response(client):
    response = client.get("/about")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "About the Academic Project" in html
    assert "Development Lifecycle (Phases 1–7)" in html
    assert "Academic Research Disclaimer" in html


def test_invalid_file_rejection_no_file(client):
    response = client.post("/predict", data={}, follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "No image file provided" in html or "No image selected" in html


def test_unsupported_format_rejection(client):
    data = {
        "image": (io.BytesIO(b"dummy text content"), "report.txt"),
        "modality": "eye_conjunctiva",
        "model_id": "resnet18"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Unsupported file format" in html


def test_corrupt_image_rejection(client):
    data = {
        "image": (io.BytesIO(b"not an image at all but with png extension"), "fake.png"),
        "modality": "eye_conjunctiva",
        "model_id": "resnet18"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Corrupted or unreadable image" in html


def test_unsupported_modality_or_model(client, real_eye_image_bytes):
    img_bytes, filename = real_eye_image_bytes
    # Deep learning requested on palm (unsupported)
    data = {
        "image": (io.BytesIO(img_bytes), filename),
        "modality": "palmar_pallor",
        "model_id": "resnet18"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "not available" in html or "Invalid" in html


def test_deep_learning_custom_cnn_inference_and_cam(client, real_eye_image_bytes):
    img_bytes, filename = real_eye_image_bytes
    data = {
        "image": (io.BytesIO(img_bytes), filename),
        "modality": "eye_conjunctiva",
        "model_id": "custom_cnn"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Inference & Explainability Results" in html
    assert "Model prediction:" in html
    assert "Confidence Score:" in html
    assert "gradcam_overlay.png" in html
    assert "layercam_overlay.png" in html
    assert "guided_backprop.png" in html
    assert "guided_gradcam.png" in html
    assert "Academic Prototype Notice:" in html


def test_deep_learning_resnet18_inference_and_cam(client, real_eye_image_bytes):
    img_bytes, filename = real_eye_image_bytes
    data = {
        "image": (io.BytesIO(img_bytes), filename),
        "modality": "eye_conjunctiva",
        "model_id": "resnet18"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Inference & Explainability Results" in html
    assert "Pretrained ResNet-18" in html
    assert "Model prediction:" in html


def test_classical_ml_random_forest_and_shap(client, real_eye_image_bytes):
    img_bytes, filename = real_eye_image_bytes
    data = {
        "image": (io.BytesIO(img_bytes), filename),
        "modality": "eye_conjunctiva",
        "model_id": "random_forest"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Inference & Explainability Results" in html
    assert "Random Forest" in html
    assert "Model prediction:" in html
    assert "shap_waterfall.png" in html
    assert "Key Feature Contributions:" in html


def test_classical_ml_svm_and_shap(client, real_eye_image_bytes):
    img_bytes, filename = real_eye_image_bytes
    data = {
        "image": (io.BytesIO(img_bytes), filename),
        "modality": "eye_conjunctiva",
        "model_id": "svm"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Inference & Explainability Results" in html
    assert "Support Vector Machine" in html
    assert "Model prediction:" in html
    assert "shap_waterfall.png" in html


def test_classical_ml_palm_modality(client, real_palm_image_bytes):
    img_bytes, filename = real_palm_image_bytes
    data = {
        "image": (io.BytesIO(img_bytes), filename),
        "modality": "palmar_pallor",
        "model_id": "logistic_regression"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Inference & Explainability Results" in html
    assert "Logistic Regression" in html
    assert "Palmar Pallor" in html


def test_error_handling_404(client):
    response = client.get("/invalid_page_404")
    assert response.status_code == 404
    html = response.data.decode("utf-8")
    assert "Error 404" in html
    assert "Page Not Found" in html
