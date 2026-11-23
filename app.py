"""
Flask Web Application for AI-Based Anemia Detection
Integrates trained Classical ML and Deep Learning models with real-time Explainable AI (SHAP & LayerCAM).

Author: Senior Machine Learning & Deep Learning Engineer
Project: AI-Based Anemia Detection Using Image Processing, ML, DL, and XAI
"""

import os
import sys
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from flask import (
    Flask, render_template, request, redirect, url_for, flash, abort, jsonify
)
from werkzeug.utils import secure_filename
from PIL import Image

# Ensure workspace root is in path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config import Config
from src.inference_service import (
    load_model,
    extract_image_features,
    predict_classical_ml,
    predict_deep_learning,
    explain_deep_learning,
    explain_classical_ml
)
import pandas as pd

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload and generated folders exist
Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
Config.GENERATED_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def is_valid_image(file_stream) -> bool:
    try:
        file_stream.seek(0)
        with Image.open(file_stream) as img:
            img.verify()
        file_stream.seek(0)
        return True
    except Exception:
        return False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Validate file presence
    if "image" not in request.files:
        flash("No image file provided in request.", "error")
        return redirect(url_for("prediction"))

    file = request.files["image"]
    if file.filename == "":
        flash("No image selected. Please choose a valid image file.", "error")
        return redirect(url_for("prediction"))

    if not allowed_file(file.filename):
        flash("Unsupported file format! Please upload a valid JPG, JPEG, or PNG image.", "error")
        return redirect(url_for("prediction"))

    if not is_valid_image(file.stream):
        flash("Corrupted or unreadable image file! Please choose a valid image.", "error")
        return redirect(url_for("prediction"))

    # 2. Validate modality & model selection
    modality = request.form.get("modality", "eye_conjunctiva")
    model_id = request.form.get("model_id", "resnet18")

    if modality not in Config.SUPPORTED_MODALITIES:
        flash(f"Invalid modality selected: '{modality}'", "error")
        return redirect(url_for("prediction"))

    modality_info = Config.SUPPORTED_MODALITIES[modality]
    if model_id not in modality_info["models"]:
        flash(f"Model '{model_id}' is not available for modality '{modality_info['display_name']}'.", "error")
        return redirect(url_for("prediction"))

    model_metadata = modality_info["models"][model_id]
    model_name = model_metadata["name"]
    model_type = model_metadata["type"]

    # 3. Save uploaded image safely
    prediction_id = str(uuid.uuid4())[:12]
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pred_gen_dir = Config.GENERATED_FOLDER / prediction_id
    pred_gen_dir.mkdir(parents=True, exist_ok=True)

    orig_ext = file.filename.rsplit(".", 1)[1].lower()
    saved_img_name = f"input_image.{orig_ext}"
    saved_img_path = pred_gen_dir / saved_img_name

    try:
        file.save(saved_img_path)
    except Exception as e:
        flash(f"Failed to save uploaded file securely: {e}", "error")
        return redirect(url_for("prediction"))

    # 4. Model Inference & Explainability
    try:
        model = load_model(modality, model_id)
    except Exception as e:
        flash(f"Error loading model '{model_name}': {e}", "error")
        return redirect(url_for("prediction"))

    try:
        dl_xai = None
        ml_xai = None

        if model_type == "Deep Learning":
            pred_res = predict_deep_learning(model, saved_img_path, model_id)
            # Generate DL XAI
            dl_xai = explain_deep_learning(
                model=model,
                model_id=model_id,
                image_path=saved_img_path,
                output_subdir=pred_gen_dir,
                target_class=pred_res["class_index"]
            )
        else:
            features_vec, feature_names = extract_image_features(saved_img_path)
            pred_res = predict_classical_ml(model, features_vec, model_id)
            # Generate ML SHAP XAI
            if modality_info["models"][model_id].get("supports_shap", False):
                ml_xai = explain_classical_ml(
                    model=model,
                    model_id=model_id,
                    features_vec=features_vec,
                    output_subdir=pred_gen_dir,
                    pred_info=pred_res
                )

        uploaded_rel_path = f"generated/{prediction_id}/{saved_img_name}"

        return render_template(
            "results.html",
            prediction_id=prediction_id,
            timestamp=timestamp_str,
            result=pred_res,
            model_name=model_name,
            model_type=model_type,
            modality_display=modality_info["display_name"],
            uploaded_image_rel=uploaded_rel_path,
            dl_xai=dl_xai,
            ml_xai=ml_xai
        )

    except Exception as e:
        flash(f"An error occurred during prediction processing: {e}", "error")
        return redirect(url_for("prediction"))


@app.route("/dashboard")
def dashboard():
    metrics_path = Config.PHASE5_RESULTS / "final_test_metrics.csv"
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        test_metrics = df_metrics.to_dict(orient="records")
    else:
        test_metrics = []

    return render_template("dashboard.html", test_metrics=test_metrics)


@app.route("/model_comparison")
def model_comparison():
    return render_template("model_comparison.html")


@app.route("/explainable_ai")
def explainable_ai():
    return render_template("explainable_ai.html")


@app.route("/about")
def about():
    return render_template("about.html")


# Error Handlers
@app.errorhandler(400)
def bad_request_error(e):
    return render_template("error.html", error_code=400, error_title="Bad Request", error_message="The request could not be understood or was missing required parameters."), 400


@app.errorhandler(404)
def not_found_error(e):
    return render_template("error.html", error_code=404, error_title="Page Not Found", error_message="The requested resource or page could not be located on the server."), 404


@app.errorhandler(413)
def request_entity_too_large(e):
    return render_template("error.html", error_code=413, error_title="Payload Too Large", error_message="Uploaded image exceeds the maximum allowed file size of 16 MB. Please upload a smaller image."), 413


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", error_code=500, error_title="Internal Server Error", error_message="An unexpected error occurred while processing your request. Please try again or return home."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"\n==================================================================")
    print(f"ANEMIAScan AI WEB APPLICATION RUNNING")
    print(f"URL: http://127.0.0.1:{port}/")
    print(f"==================================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
