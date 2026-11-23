"""
Reusable PyTorch Training Engine for Deep Learning Models
Author: Senior Deep Learning Engineer & PyTorch Developer
Project: AI-Based Anemia Detection
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Detects available hardware device."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        num_threads = min(os.cpu_count() or 4, 10)
        torch.set_num_threads(num_threads)
        print(f"Device: CPU (Configured {num_threads} CPU execution threads)")
    return device


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float = 1.0
) -> tuple[float, float]:
    """Trains model for a single epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Check for NaN / Inf loss values
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(f"[CRITICAL ERROR] Encountered NaN or Inf loss: {loss.item()}")

        loss.backward()
        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates model on validation dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    return epoch_loss, epoch_acc, bal_acc, roc_auc, f1_macro, y_true, y_pred, y_prob


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 12,
    patience: int = 4,
    checkpoint_path: Optional[Path] = None,
    model_name: str = "Model",
    modality_name: str = "Eye"
) -> tuple[nn.Module, dict, dict]:
    """
    Executes training loop with Early Stopping, Checkpointing, and Metric Logging.
    """
    set_seed(42)
    device = get_device()
    model = model.to(device)

    # Class-weighted loss
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_bal_acc": [], "val_roc_auc": [],
        "val_macro_f1": []
    }

    best_score = -1.0
    best_epoch = 0
    best_metrics = {}
    epochs_no_improve = 0

    print(f"\nStarting Training: {model_name} on {modality_name} ({num_epochs} Epochs, Early Stopping Patience={patience})", flush=True)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_bal_acc, v_auc, v_f1, y_true, y_pred, y_prob = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(v_auc)

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))
        history["val_bal_acc"].append(round(v_bal_acc, 4))
        history["val_roc_auc"].append(round(v_auc, 4))
        history["val_macro_f1"].append(round(v_f1, 4))

        elapsed = time.time() - ep_start
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {t_loss:.4f}, Acc: {t_acc:.4f} | "
            f"Val Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, BalAcc: {v_bal_acc:.4f}, AUC: {v_auc:.4f}, F1: {v_f1:.4f}",
            flush=True
        )

        # Monitor ROC-AUC / Balanced Accuracy composite for model checkpointing
        composite_score = v_auc
        if composite_score > best_score:
            best_score = composite_score
            best_epoch = epoch
            best_metrics = {
                "epoch": epoch,
                "train_loss": t_loss, "train_acc": t_acc,
                "val_loss": v_loss, "val_acc": v_acc,
                "val_bal_acc": v_bal_acc, "val_roc_auc": v_auc,
                "val_macro_f1": v_f1,
                "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob
            }
            epochs_no_improve = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "val_metrics": {k: float(v) for k, v in best_metrics.items() if isinstance(v, (int, float))}
                }, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [Early Stopping Triggered] No improvement in validation score for {patience} consecutive epochs.", flush=True)
                break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s. Best Epoch: {best_epoch} with Val AUC: {best_score:.4f}", flush=True)

    # Load best checkpoint weights
    if checkpoint_path is not None and checkpoint_path.exists():
        chk = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(chk["model_state_dict"])

    best_metrics["total_training_time_seconds"] = round(total_time, 2)
    return model, history, best_metrics
