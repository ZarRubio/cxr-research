"""
src/training/train_s4.py

Loop de entrenamiento Sprint 4 — CNN-ViT híbrido, 2 fases.

Fase 1 (CNN congelada):
  - Solo se entrenan los parámetros del ViT y la cabeza clasificadora.
  - LR alto (1e-4): el ViT aprende rápido a usar las features de la CNN.
  - Epochs: 15, patience: 5.

Fase 2 (full fine-tuning con LR diferenciado):
  - CNN: lr_cnn=1e-6 (muy conservador, no destruir preentrenamiento).
  - ViT: lr_vit=1e-5 (sigue aprendiendo pero más despacio que en fase 1).
  - Epochs: 30, patience: 8.

El checkpoint final guarda el mejor modelo de la FASE 2.
El checkpoint de fase 1 se guarda por separado (útil para ablation study).
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
    compute_metrics_multiclass,
    logits_to_probs_multiclass,
)
from src.training.utils import save_checkpoint


# ---------------------------------------------------------------------------
# Calcular class weights
# ---------------------------------------------------------------------------
def compute_class_weights(loader: DataLoader, num_classes: int,
                          device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for batch in loader:
        for label in batch["label"]:
            counts[label.item()] += 1
    total   = counts.sum()
    weights = total / (num_classes * counts)
    return (weights / weights.sum() * num_classes).to(device)


# ---------------------------------------------------------------------------
# Una epoch de entrenamiento
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    use_amp: bool,
                    scaler) -> dict:
    model.train()
    losses, all_probs, all_labels = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
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
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return {
        "loss":    float(np.mean(losses)),
        "metrics": compute_metrics_multiclass(all_labels, all_probs),
    }


# ---------------------------------------------------------------------------
# Una epoch de validación
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader,
                       criterion: nn.Module,
                       device: torch.device) -> dict:
    model.eval()
    losses, all_probs, all_labels = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        losses.append(criterion(logits, labels).item())
        probs = logits_to_probs_multiclass(logits.float())
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return {
        "loss":    float(np.mean(losses)),
        "metrics": compute_metrics_multiclass(all_labels, all_probs),
    }


# ---------------------------------------------------------------------------
# Loop de una fase
# ---------------------------------------------------------------------------
def run_phase(model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, optimizer: torch.optim.Optimizer,
              criterion: nn.Module, scheduler, epochs: int,
              patience: int, device: torch.device,
              use_amp: bool, logger, phase_name: str,
              ckpt_path: Path) -> tuple[dict, list]:
    """
    Corre una fase de entrenamiento completa.
    Returns: (mejor estado del modelo, historial de epochs)
    """
    scaler = GradScaler(device="cuda") if use_amp else None

    best_auc   = -float("inf")
    best_state = None
    no_improve = 0
    history    = []

    logger.info("─" * 65)
    logger.info("FASE %s | %d epochs | patience=%d", phase_name, epochs, patience)
    logger.info("Parámetros entrenables: %d", model.trainable_params())
    logger.info("─" * 65)

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_loader, optimizer,
                             criterion, device, use_amp, scaler)
        va = validate_one_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(va["loss"])

        logger.info(
            "[%s] Epoch %2d/%d | LR=%.2e | "
            "train loss=%.4f AUC=%.4f | val loss=%.4f AUC=%.4f acc=%.4f",
            phase_name, epoch, epochs, lr,
            tr["loss"], tr["metrics"]["auc_macro"],
            va["loss"], va["metrics"]["auc_macro"], va["metrics"]["accuracy"],
        )
        for i, auc in enumerate(va["metrics"]["auc_per_class"]):
            logger.info("  val AUC clase %d: %.4f", i, auc)

        history.append({
            "phase": phase_name, "epoch": epoch, "lr": lr,
            "train_loss": tr["loss"], "train_metrics": tr["metrics"],
            "val_loss":   va["loss"], "val_metrics":   va["metrics"],
        })

        score = va["metrics"]["auc_macro"]
        if score > best_auc:
            best_auc   = score
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            logger.info("  → Nuevo mejor (val AUC_macro=%.4f)", best_auc)
            # Guardar checkpoint de esta fase
            save_checkpoint({
                "phase": phase_name, "epoch": epoch,
                "model_state_dict": best_state,
                "best_val_auc_macro": best_auc,
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  Early stopping en %s (epoch %d).", phase_name, epoch)
                break

    logger.info("Fase %s terminada. Mejor AUC_macro=%.4f", phase_name, best_auc)
    return best_state, history


# ---------------------------------------------------------------------------
# Loop principal — 2 fases
# ---------------------------------------------------------------------------
def train_model_cnn_vit(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        cfg: Any,
                        device: torch.device,
                        logger) -> dict:
    """
    Entrena el CNN-ViT en 2 fases.

    Fase 1: CNN congelada, solo ViT + head.
    Fase 2: Todo desbloqueado, LR diferenciado CNN vs ViT.
    """
    ckpt_dir = Path(cfg.paths["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = cfg.training.get("use_amp", False) and torch.cuda.is_available()
    use_cw  = cfg.training.get("use_class_weights", True)

    # Class weights (calculados una vez)
    if use_cw:
        logger.info("Calculando class weights...")
        weights   = compute_class_weights(train_loader, cfg.model["num_classes"], device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info("Class weights: %s", weights.tolist())
    else:
        criterion = nn.CrossEntropyLoss()

    p1_cfg = cfg.training["phase1"]
    p2_cfg = cfg.training["phase2"]
    all_history = []

    # ─── FASE 1 ──────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("INICIANDO FASE 1: CNN congelada, entrenando solo ViT")
    logger.info("=" * 65)

    model.set_phase(1)
    logger.info("Arquitectura fase 1: %s", model.param_summary())

    opt1 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=p1_cfg["learning_rate"],
        weight_decay=p1_cfg.get("weight_decay", 1e-4),
    )

    best_state_p1, history_p1 = run_phase(
        model, train_loader, val_loader, opt1, criterion,
        scheduler=None,
        epochs=p1_cfg["epochs"],
        patience=p1_cfg.get("patience", 5),
        device=device, use_amp=use_amp, logger=logger,
        phase_name="P1",
        ckpt_path=ckpt_dir / "sprint4_phase1_best.pt",
    )
    all_history.extend(history_p1)

    # Cargar mejor estado de fase 1 antes de iniciar fase 2
    if best_state_p1 is not None:
        model.load_state_dict(best_state_p1)
    best_p1_auc = max(h["val_metrics"]["auc_macro"] for h in history_p1)

    # ─── FASE 2 ──────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("INICIANDO FASE 2: full fine-tuning con LR diferenciado")
    logger.info("  CNN LR: %.2e | ViT LR: %.2e",
                p2_cfg["lr_cnn"], p2_cfg["lr_vit"])
    logger.info("=" * 65)

    model.set_phase(2)
    logger.info("Arquitectura fase 2: %s", model.param_summary())

    param_groups = model.get_param_groups(
        lr_cnn=p2_cfg["lr_cnn"],
        lr_vit=p2_cfg["lr_vit"],
    )
    opt2 = torch.optim.Adam(param_groups,
                             weight_decay=p2_cfg.get("weight_decay", 1e-4))

    sched_cfg = p2_cfg.get("scheduler", {})
    scheduler2 = None
    if sched_cfg.get("enabled", False):
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt2,
            mode=sched_cfg.get("mode", "min"),
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 3),
            min_lr=sched_cfg.get("min_lr", 1e-7),
        )

    best_state_p2, history_p2 = run_phase(
        model, train_loader, val_loader, opt2, criterion,
        scheduler=scheduler2,
        epochs=p2_cfg["epochs"],
        patience=p2_cfg.get("patience", 8),
        device=device, use_amp=use_amp, logger=logger,
        phase_name="P2",
        ckpt_path=ckpt_dir / "sprint4_phase2_best.pt",
    )
    all_history.extend(history_p2)

    best_p2_auc = max(h["val_metrics"]["auc_macro"] for h in history_p2)
    best_p2_epoch = next(
        h["epoch"] for h in reversed(history_p2)
        if h["val_metrics"]["auc_macro"] == best_p2_auc
    )

    logger.info("=" * 65)
    logger.info("ENTRENAMIENTO CNN-ViT TERMINADO")
    logger.info("  Fase 1 best AUC_macro: %.4f", best_p1_auc)
    logger.info("  Fase 2 best AUC_macro: %.4f", best_p2_auc)
    logger.info("  Checkpoint final: %s", ckpt_dir / "sprint4_phase2_best.pt")
    logger.info("=" * 65)

    return {
        "history":            all_history,
        "best_epoch_phase1":  max(range(len(history_p1)),
                                   key=lambda i: history_p1[i]["val_metrics"]["auc_macro"]) + 1,
        "best_epoch_phase2":  best_p2_epoch,
        "best_val_auc_phase1": best_p1_auc,
        "best_val_auc_phase2": best_p2_auc,
        "checkpoint_path":    str(ckpt_dir / "sprint4_phase2_best.pt"),
    }
