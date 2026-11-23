"""
Custom Convolutional Neural Network (CNN) for Anemia Detection
Architecture: 4-stage Conv-BatchNorm-ReLU-MaxPool with Dropout and Adaptive Pooling.
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

import torch
import torch.nn as nn


class AnemiaCNN(nn.Module):
    """
    Lightweight, highly regularized 4-stage CNN architecture for medical image classification.
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super(AnemiaCNN, self).__init__()

        # Block 1: 3 -> 32 (224x224 -> 112x112)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 32 -> 64 (112x112 -> 56x56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: 64 -> 128 (56x56 -> 28x28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: 128 -> 256 (28x28 -> 14x14)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Pooling and Regularized Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


def get_custom_cnn(num_classes: int = 2, dropout_rate: float = 0.4) -> AnemiaCNN:
    """Helper factory for AnemiaCNN."""
    return AnemiaCNN(num_classes=num_classes, dropout_rate=dropout_rate)
