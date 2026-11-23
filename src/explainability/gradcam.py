"""
PyTorch Grad-CAM (Gradient-Weighted Class Activation Mapping) Implementation
Computes class-discriminative spatial localization heatmaps from convolutional feature layers.

Author: Senior Machine Learning Engineer & Explainable AI Specialist
Project: AI-Based Anemia Detection
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM engine utilizing forward and backward PyTorch hooks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h_f = self.target_layer.register_forward_hook(forward_hook)
        h_b = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h_f, h_b])

    def generate_heatmap(self, input_tensor: torch.Tensor,
                         target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Generates 2D normalized Grad-CAM heatmap for a single input tensor (1, C, H, W).
        Returns: (heatmap_np [H, W], predicted_class, predicted_probability)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        pred_prob = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        # Backward pass on the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients: weights = (C, 1, 1)
        gradients = self.gradients[0]  # (C, H_act, W_act)
        activations = self.activations[0]  # (C, H_act, W_act)

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)  # (H_act, W_act)

        # ReLU non-linearity: retain features that have positive impact on target class
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()

        # Normalize to [0, 1] range
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, pred_class, pred_prob

    def remove_hooks(self):
        """Removes registered PyTorch hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
