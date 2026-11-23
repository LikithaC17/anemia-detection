"""
Guided Backpropagation and Guided Grad-CAM Implementation
Combines pixel-level high-frequency gradient saliency with class-discriminative coarse heatmaps.
Reference: Springenberg et al., 'Striving for Simplicity: The All Convolutional Net' (ICLR 2015)
           Selvaraju et al., 'Grad-CAM: Visual Explanations from Deep Networks' (ICCV 2017)

Author: Senior Deep Learning Engineer & Explainable AI Specialist
Project: AI-Based Anemia Detection
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2


class GuidedBackprop:
    """
    Guided Backpropagation engine utilizing custom backward hooks on all ReLU activations.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def relu_backward_hook(module, grad_in, grad_out):
            # Clamp negative gradient components during backpropagation
            return (torch.clamp(grad_in[0], min=0.0),)

        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                # Ensure inplace is False so backward hooks operate correctly
                m.inplace = False
                self.hooks.append(m.register_full_backward_hook(relu_backward_hook))

    def generate_saliency(self, input_tensor: torch.Tensor,
                          target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Computes guided backpropagation gradients w.r.t input image pixels.
        Returns:
            rgb_saliency: (H, W, 3) normalized RGB saliency visualization
            gray_saliency: (H, W) normalized 2D intensity map
            pred_class: int
            pred_prob: float
        """
        self.model.eval()
        img_var = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(img_var)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        pred_prob = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        grad = img_var.grad[0].cpu().numpy()  # (3, H, W)
        grad = np.transpose(grad, (1, 2, 0))   # (H, W, 3)

        # 2D Saliency Magnitude across color channels
        gray_saliency = np.max(np.abs(grad), axis=2)
        g_min, g_max = gray_saliency.min(), gray_saliency.max()
        if g_max - g_min > 1e-8:
            gray_saliency = (gray_saliency - g_min) / (g_max - g_min)
        else:
            gray_saliency = np.zeros_like(gray_saliency)

        # 3D RGB Saliency
        rgb_saliency = np.abs(grad)
        r_min, r_max = rgb_saliency.min(), rgb_saliency.max()
        if r_max - r_min > 1e-8:
            rgb_saliency = (rgb_saliency - r_min) / (r_max - r_min)
        else:
            rgb_saliency = np.zeros_like(rgb_saliency)

        return rgb_saliency, gray_saliency, pred_class, pred_prob

    def generate_guided_gradcam(self, input_tensor: torch.Tensor,
                                gradcam_heatmap: np.ndarray,
                                target_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Point-wise fuses high-resolution Guided Backprop saliency with coarse Grad-CAM heatmap.
        Returns:
            guided_gradcam_rgb: (H, W, 3) in [0, 1]
            guided_gradcam_gray: (H, W) in [0, 1]
        """
        rgb_sal, gray_sal, _, _ = self.generate_saliency(input_tensor, target_class=target_class)

        h, w = gray_sal.shape
        heatmap_resized = cv2.resize(gradcam_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        # Pointwise multiplication
        guided_gradcam_gray = gray_sal * heatmap_resized
        guided_gradcam_rgb = rgb_sal * np.expand_dims(heatmap_resized, axis=2)

        # Normalize to [0, 1]
        g_min, g_max = guided_gradcam_gray.min(), guided_gradcam_gray.max()
        if g_max - g_min > 1e-8:
            guided_gradcam_gray = (guided_gradcam_gray - g_min) / (g_max - g_min)
        else:
            guided_gradcam_gray = np.zeros_like(guided_gradcam_gray)

        r_min, r_max = guided_gradcam_rgb.min(), guided_gradcam_rgb.max()
        if r_max - r_min > 1e-8:
            guided_gradcam_rgb = (guided_gradcam_rgb - r_min) / (r_max - r_min)
        else:
            guided_gradcam_rgb = np.zeros_like(guided_gradcam_rgb)

        return guided_gradcam_rgb, guided_gradcam_gray

    def remove_hooks(self):
        """Removes registered PyTorch hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
