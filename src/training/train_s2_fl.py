"""
src/training/train_s2_fl.py
Loop de entrenamiento Sprint 2 con Focal Loss.
Basado en train_s2.py, con FocalLoss en lugar de CrossEntropyLoss.
"""
from __future__ import annotations
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.evaluation.metrics_s2 import (
    compute_metrics_multiclass, logits_to_probs_multiclass
)
from src.training.focal_loss import FocalLoss
from src.training.utils import save_checkpoint


def compute_alpha(train_loader, num_classes, device):
    """Alpha inversamente proporcional a la frecuencia de clase."""
    counts = torch.zeros(num_classes)
    for batch in train_loader:
        for label in batch["label"]:
            counts[label.item()] += 1
    total  = counts.sum()
    alpha  = total / (num_classes * counts)
    alpha  = alpha / alpha.sum() * num_classes
    return alpha.to(device)


def train_one_epoch_fl(model, loader, optimizer, criterion,
                       device, use_amp, scaler):
    model.train()
    losses, all_probs, all_labels = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            probs = logits_to_probs_multiclass(logits.detach().float())
            all_probs.append(probs.cpu()); all_labels.append(labels.cpu())
    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return {"loss": float(np.mean(losses)),
            "metrics": compute_metrics_multiclass(all_labels, all_probs)}


@torch.no_grad()
def validate_one_epoch_fl(model, loader, criterion, device):
    model.eval()
    losses, all_probs, all_labels = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(criterion(logits, labels).item())
        probs = logits_to_probs_multiclass(logits.float())
        all_probs.append(probs.cpu()); all_labels.append(labels.cpu())
    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return {"loss": float(np.mean(losses)),
            "metrics": compute_metrics_multiclass(all_labels, all_probs)}


def train_model_focal(model, train_loader, val_loader, cfg, device, logger):
    num_classes = cfg.model["num_classes"]
    trainable   = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.Adam(
        trainable, lr=cfg.training["learning_rate"],
        weight_decay=cfg.training.get("weight_decay", 0.0)
    )
    logger.info("Calculando alpha para Focal Loss...")
    alpha    = compute_alpha(train_loader, num_classes, device)
    logger.info("Alpha: %s", alpha.tolist())
    criterion = FocalLoss(gamma=2.0, alpha=alpha)
    logger.info("Focal Loss (gamma=2.0) activado.")

    use_amp  = cfg.training.get("use_amp", False) and torch.cuda.is_available()
    scaler   = GradScaler(device="cuda") if use_amp else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    epochs, patience = cfg.training["epochs"], cfg.training.get("patience", 8)
    ckpt_dir = Path(cfg.paths["checkpoints"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "sprint2_focal_best.pt"

    best_auc, best_epoch, best_state = -float("inf"), -1, None
    no_improve, history = 0, []

    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento con Focal Loss: %d epochs", epochs)
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch_fl(model, train_loader, optimizer,
                                criterion, device, use_amp, scaler)
        va = validate_one_epoch_fl(model, val_loader, criterion, device)
        scheduler.step(va["loss"])

        logger.info(
            "Epoch %2d/%d | LR=%.2e | train loss=%.4f AUC=%.4f | "
            "val loss=%.4f AUC=%.4f acc=%.4f",
            epoch, epochs, lr,
            tr["loss"], tr["metrics"]["auc_macro"],
            va["loss"], va["metrics"]["auc_macro"], va["metrics"]["accuracy"],
        )
        for i, auc in enumerate(va["metrics"]["auc_per_class"]):
            logger.info("  val AUC clase %d: %.4f", i, auc)

        history.append({
            "epoch": epoch, "lr": lr,
            "train_loss": tr["loss"], "train_metrics": tr["metrics"],
            "val_loss":   va["loss"], "val_metrics":   va["metrics"],
        })

        score = va["metrics"]["auc_macro"]
        if score > best_auc:
            best_auc, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            logger.info("  → Nuevo mejor (AUC_macro=%.4f)", best_auc)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping.")
                break

    if best_state:
        save_checkpoint({
            "epoch": best_epoch, "model_state_dict": best_state,
            "best_val_auc_macro": best_auc,
            "loss": "FocalLoss", "gamma": 2.0,
            "cfg_snapshot": {
                "backbone": cfg.model["backbone"],
                "num_classes": cfg.model["num_classes"],
                "finetune_mode": cfg.model["finetune_mode"],
            },
        }, best_ckpt)
        logger.info("Mejor modelo guardado: %s (epoch %d, AUC=%.4f)",
                    best_ckpt, best_epoch, best_auc)

    logger.info("Entrenamiento Focal Loss terminado.")
    return {"history": history, "best_epoch": best_epoch,
            "best_val_auc_macro": best_auc, "checkpoint_path": str(best_ckpt)}
