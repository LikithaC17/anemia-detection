"""
Grad-CAM Explanations Module for Custom CNN Architecture
Target layer: Conv2d of Block 4 (256 feature channels)

Author: Senior Deep Learning Engineer & Explainable AI Specialist
Project: AI-Based Anemia Detection
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.deep_learning.custom_cnn import get_custom_cnn
from src.deep_learning.transforms import get_val_transforms
from src.explainability.gradcam import GradCAM
from src.explainability.explanation_utils import denormalize_tensor, apply_colormap_on_image


def run_custom_cnn_gradcam(checkpoint_path: Path,
                           test_manifest_path: Path,
                           output_base_dir: Path,
                           samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Executes Grad-CAM for the trained Custom CNN on representative test samples.
    """
    device = torch.device("cpu")
    torch.set_num_threads(8)

    # 1. Load model and checkpoint
    model = get_custom_cnn(num_classes=2)
    chk = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(chk["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Select target layer: final convolution block (block4[0])
    target_layer = model.block4[0]
    gradcam = GradCAM(model, target_layer)

    orig_dir = output_base_dir / "original_images"
    heat_dir = output_base_dir / "heatmaps"
    over_dir = output_base_dir / "overlays"
    for d in [orig_dir, heat_dir, over_dir]:
        d.mkdir(parents=True, exist_ok=True)

    transforms_pipeline = get_val_transforms()
    records = []

    gallery_items = []

    print(f"Running Grad-CAM on Custom CNN for {len(samples)} representative samples...")

    for i, s in enumerate(samples):
        img_path = Path(s["image_path"])
        if not img_path.exists():
            print(f"[WARNING] Image path {img_path} does not exist!")
            continue

        with Image.open(img_path) as raw_img:
            raw_rgb = raw_img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)

        input_tensor = transforms_pipeline(raw_rgb).unsqueeze(0).to(device)

        # Target class: true class or predicted class
        pred_logits = model(input_tensor)
        pred_idx = torch.argmax(pred_logits, dim=1).item()
        prob_anemia = torch.softmax(pred_logits, dim=1)[0, 1].item()

        # Generate heatmap targeting the model's prediction
        heatmap_2d, _, _ = gradcam.generate_heatmap(input_tensor, target_class=pred_idx)

        # Denormalize input tensor to get exact RGB image seen by the network
        rgb_disp = denormalize_tensor(input_tensor)
        colored_hm, overlay = apply_colormap_on_image(rgb_disp, heatmap_2d, alpha=0.50)

        # Save individual images
        file_slug = f"sample_{i+1:02d}_{s['category'].split()[0].lower()}_{pred_idx}"
        orig_p = orig_dir / f"{file_slug}_original.png"
        heat_p = heat_dir / f"{file_slug}_heatmap.png"
        over_p = over_dir / f"{file_slug}_overlay.png"

        Image.fromarray(rgb_disp).save(orig_p)
        Image.fromarray(colored_hm).save(heat_p)
        Image.fromarray(overlay).save(over_p)

        pred_label_str = "Anemia" if pred_idx == 1 else "Non_Anemia"
        rec = {
            "sample_id": i + 1,
            "model_name": "Custom CNN",
            "modality": "Eye Conjunctiva",
            "category": s["category"],
            "image_path": str(img_path),
            "subject_id": s.get("subject_id", "Unknown"),
            "true_label": s["true_label"],
            "predicted_label": pred_label_str,
            "prob_anemia": round(float(prob_anemia), 4),
            "predicted_confidence": round(float(prob_anemia if pred_idx == 1 else 1.0 - prob_anemia), 4),
            "target_layer": "block4.0 (Conv2d: 128 -> 256)",
            "original_image_path": str(orig_p),
            "heatmap_path": str(heat_p),
            "overlay_path": str(over_p),
            "interpretation_summary": s.get("notes", "")
        }
        records.append(rec)
        gallery_items.append((rgb_disp, colored_hm, overlay, rec))

    gradcam.remove_hooks()

    # Generate Consolidated Visual Gallery
    n_items = len(gallery_items)
    if n_items > 0:
        fig, axes = plt.subplots(n_items, 3, figsize=(11, 3.2 * n_items))
        if n_items == 1:
            axes = np.expand_dims(axes, 0)

        for row, (orig_img, h_map, ov_img, r) in enumerate(gallery_items):
            axes[row, 0].imshow(orig_img)
            axes[row, 0].set_title(f"Original: {r['category']}\nTrue: {r['true_label']} | Subject: {r['subject_id']}", fontsize=9, fontweight="bold")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(h_map)
            axes[row, 1].set_title(f"Grad-CAM Heatmap (Block 4)\nTarget: {r['predicted_label']}", fontsize=9, fontweight="bold")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(ov_img)
            col = "green" if r["true_label"] == r["predicted_label"] else "red"
            axes[row, 2].set_title(f"Overlay | Pred: {r['predicted_label']} (P={r['prob_anemia']:.2f})\nConf: {r['predicted_confidence']:.2f}",
                                   fontsize=9, color=col, fontweight="bold")
            axes[row, 2].axis("off")

        plt.suptitle("Custom CNN Grad-CAM Diagnostic Visualizations (Eye Conjunctiva)", fontsize=13, fontweight="bold", y=1.001)
        plt.tight_layout()
        gallery_p = output_base_dir / "custom_cnn_gradcam_gallery.png"
        plt.savefig(gallery_p, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved Custom CNN Grad-CAM Gallery -> {gallery_p}")

    return records
