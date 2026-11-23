"""
ResNet-18 Transfer Learning Model for Anemia Detection
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet18_transfer_model(
    num_classes: int = 2,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.4
) -> nn.Module:
    """
    Constructs a pretrained ResNet-18 with adapted classification head.
    
    Parameters:
        num_classes: Number of target categories (default: 2 for binary Anemia/Non_Anemia).
        freeze_backbone: If True, freezes conv1 through layer3 to retain ImageNet representations.
                         layer4 and fc head remain trainable for domain adaptation.
        dropout_rate: Dropout probability in the classification head.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        # Freeze early and mid-level feature extractors (conv1, bn1, layer1, layer2, layer3)
        for name, param in model.named_parameters():
            if not name.startswith("layer4") and not name.startswith("fc"):
                param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model
