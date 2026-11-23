"""
Unit and Integration Tests for Phase 4: Deep Learning Implementation & Verification
Author: Senior Deep Learning Engineer & PyTorch Specialist
Project: AI-Based Anemia Detection
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image

WORKSPACE_ROOT = Path(r"c:\Users\LIKITHA\OneDrive\Desktop\Anemia Detection")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.deep_learning.custom_cnn import AnemiaCNN, get_custom_cnn
from src.deep_learning.resnet18_model import get_resnet18_transfer_model
from src.deep_learning.transforms import get_train_transforms, get_val_transforms
from src.deep_learning.dataset import AnemiaDataset


def test_custom_cnn_architecture():
    """Verify CNN layers, shapes, and forward pass."""
    model = get_custom_cnn(num_classes=2, dropout_rate=0.4)
    assert isinstance(model, AnemiaCNN)
    
    # 4 convolutional blocks
    assert hasattr(model, "block1")
    assert hasattr(model, "block2")
    assert hasattr(model, "block3")
    assert hasattr(model, "block4")
    assert hasattr(model, "classifier")
    
    # Dummy input tensor (B=4, C=3, H=224, W=224)
    x = torch.randn(4, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 2), f"Expected output shape (4, 2), got {out.shape}"


def test_resnet18_architecture_and_weights():
    """Verify ResNet-18 transfer learning, frozen backbone, and head."""
    model = get_resnet18_transfer_model(num_classes=2, freeze_backbone=True, dropout_rate=0.4)
    
    # Backbone frozen verification
    assert not model.conv1.weight.requires_grad
    assert not model.layer1[0].conv1.weight.requires_grad
    assert not model.layer2[0].conv1.weight.requires_grad
    assert not model.layer3[0].conv1.weight.requires_grad
    
    # Layer 4 and Head trainable
    assert model.layer4[0].conv1.weight.requires_grad
    assert model.fc[1].weight.requires_grad
    assert model.fc[1].out_features == 2

    # Dummy forward pass
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
    assert out.shape == (2, 2)


def test_checkpoint_loading_and_inference():
    """Verify that both saved checkpoints load cleanly and run inference."""
    cnn_path = WORKSPACE_ROOT / "models" / "deep_learning" / "cnn" / "best_custom_cnn_eye.pth"
    resnet_path = WORKSPACE_ROOT / "models" / "deep_learning" / "resnet18" / "best_resnet18_eye.pth"
    
    assert cnn_path.exists(), "CNN checkpoint missing!"
    assert resnet_path.exists(), "ResNet-18 checkpoint missing!"
    
    # 1. Custom CNN Checkpoint
    cnn_chk = torch.load(cnn_path, map_location="cpu")
    assert "model_state_dict" in cnn_chk
    assert "best_score" in cnn_chk
    cnn = get_custom_cnn(num_classes=2)
    cnn.load_state_dict(cnn_chk["model_state_dict"])
    cnn.eval()

    # 2. ResNet-18 Checkpoint
    resnet_chk = torch.load(resnet_path, map_location="cpu")
    assert "model_state_dict" in resnet_chk
    assert "best_score" in resnet_chk
    resnet = get_resnet18_transfer_model(num_classes=2, freeze_backbone=True)
    resnet.load_state_dict(resnet_chk["model_state_dict"])
    resnet.eval()

    # Synthetic sample inference
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        cnn_out = cnn(x)
        res_out = resnet(x)
        cnn_prob = torch.softmax(cnn_out, dim=1)
        res_prob = torch.softmax(res_out, dim=1)

    assert cnn_out.shape == (1, 2)
    assert res_out.shape == (1, 2)
    assert torch.isclose(cnn_prob.sum(), torch.tensor(1.0))
    assert torch.isclose(res_prob.sum(), torch.tensor(1.0))


def test_validation_image_real_inference():
    """Verify model inference on a real preprocessed validation image."""
    val_csv = WORKSPACE_ROOT / "results" / "phase2" / "validation.csv"
    assert val_csv.exists()
    
    ds = AnemiaDataset(val_csv, modality="Eye", transform=get_val_transforms())
    assert len(ds) == 397
    
    sample_tensor, label, meta = ds[0]
    assert sample_tensor.shape == (3, 224, 224)
    assert label in [0, 1]
    
    cnn_path = WORKSPACE_ROOT / "models" / "deep_learning" / "cnn" / "best_custom_cnn_eye.pth"
    cnn_chk = torch.load(cnn_path, map_location="cpu")
    cnn = get_custom_cnn(num_classes=2)
    cnn.load_state_dict(cnn_chk["model_state_dict"])
    cnn.eval()
    
    with torch.no_grad():
        logits = cnn(sample_tensor.unsqueeze(0))
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        
    assert pred in [0, 1]
    assert 0.0 <= prob[0, 1].item() <= 1.0


def test_required_artifacts_exist():
    """Verify that all Task 7 required files are present and non-empty."""
    req_files = [
        "results/phase4/phase4_final_report.txt",
        "results/phase4/deep_learning_comparison.csv",
        "results/phase4/cnn_classification_report.txt",
        "results/phase4/resnet18_classification_report.txt",
        "results/phase4/training_history.csv",
        "results/phase4/training_curves.png",
        "results/phase4/confusion_matrix_cnn.png",
        "results/phase4/confusion_matrix_resnet18.png",
        "results/phase4/checkpoints/best_custom_cnn.pth",
        "results/phase4/checkpoints/best_resnet18.pth"
    ]
    for rel_path in req_files:
        p = WORKSPACE_ROOT / rel_path
        assert p.exists(), f"Missing required file: {rel_path}"
        assert p.stat().st_size > 0, f"Empty file: {rel_path}"
