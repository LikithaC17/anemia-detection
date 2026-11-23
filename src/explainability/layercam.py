"""
LayerCAM (Layer-wise Class Activation Mapping) Implementation
Computes fine-grained class activation maps using element-wise positive gradient weighting.
Reference: Jiang et al., 'LayerCAM: Exploring Hierarchical Class Activation Maps for Localization' (IEEE TIP 2021)

Author: Senior Deep Learning Engineer & Explainable AI Specialist
Project: AI-Based Anemia Detection
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerCAM:
    """
    LayerCAM engine providing pixel-fine spatial localization without global spatial averaging.
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
        Generates 2D normalized LayerCAM heatmap for single input tensor (1, C, H, W).
        Returns: (heatmap_np [H_act, W_act], predicted_class, predicted_probability)
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        pred_prob = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        gradients = self.gradients[0]    # (C, H_act, W_act)
        activations = self.activations[0] # (C, H_act, W_act)

        # LayerCAM element-wise spatial weights: w = relu(grad)
        weights = F.relu(gradients)

        # Element-wise product and channel-wise summation: cam = relu(sum(w * act))
        cam = F.relu(torch.sum(weights * activations, dim=0))

        cam_np = cam.cpu().numpy()

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
