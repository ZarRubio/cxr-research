"""
src/training/train_multilabel.py

Loop de entrenamiento para Sprint 4 multi-label (14 clases NIH).

Diferencias clave respecto a train_s4.py (4 clases):
  - BCEWithLogitsLoss en vez de CrossEntropyLoss.
  - pos_weight para compensar el fuerte desbalance de clases.
  - Sigmoid (no softmax) en la evaluación.
  - Métricas: AUC por clase + mAP (no AUC macro de multi-clase).
  - Misma estrategia de 2 fases que Sprint 3.
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

from src.evaluation.metrics_multilabel import (
    compute_metrics_multilabel,
    format_metrics_multilabel,
)
from src.training.utils import save_checkpoint


# ─── Una epoch de entrenamiento ──────────────────────────────────
def train_one_epoch_ml(model, loader, optimizer, criterion,
                       device, use_amp, scaler):
    model.train()
    losses, all_probs, all_labels = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)  # (B, 14)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)           # (B, 14)
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
            probs = torch.sigmoid(logits.detach().float())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics    = compute_metrics_multilabel(all_labels, all_probs)
    return {"loss": float(np.mean(losses)), "metrics": metrics}


# ─── Una epoch de validación ──────────────────────────────────────
@torch.no_grad()
def validate_one_epoch_ml(model, loader, criterion, device):
    model.eval()
    losses, all_probs, all_labels = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        losses.append(criterion(logits, labels).item())
        probs = torch.sigmoid(logits.float())
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics    = compute_metrics_multilabel(all_labels, all_probs)
    return {"loss": float(np.mean(losses)), "metrics": metrics}


# ─── Loop de una fase ─────────────────────────────────────────────
def run_phase_ml(model, train_loader, val_loader, optimizer,
                 criterion, scheduler, epochs, patience,
                 device, use_amp, logger, phase_name, ckpt_path,
                 start_auc=-float("inf")):

    scaler       = GradScaler(device="cuda") if use_amp else None
    best_auc     = start_auc
    best_state   = None
    no_improve   = 0
    history      = []

    logger.info("=" * 70)
    logger.info("FASE %s | %d epochs | patience=%d", phase_name, epochs, patience)
    logger.info("Entrenables: %d", model.trainable_params())
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch_ml(model, train_loader, optimizer,
                                criterion, device, use_amp, scaler)
        va = validate_one_epoch_ml(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        logger.info(
            "[%s] Epoch %2d/%d | LR=%.2e | "
            "train loss=%.4f AUC=%.4f mAP=%.4f | "
            "val loss=%.4f AUC=%.4f mAP=%.4f",
            phase_name, epoch, epochs, lr,
            tr["loss"], tr["metrics"]["auc_macro"], tr["metrics"]["map"],
            va["loss"], va["metrics"]["auc_macro"], va["metrics"]["map"],
        )

        # Log AUC por clase en val (cada 5 epochs para no saturar el log)
        if epoch % 5 == 0 or epoch == 1:
            for i, auc in enumerate(va["metrics"]["auc_per_class"]):
                if not np.isnan(auc):
                    from src.datasets.nih_multilabel import CLASSES_14
                    logger.info("  val AUC %s: %.4f",
                                CLASSES_14[i], auc)

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
            torch.save({
                "phase": phase_name, "epoch": epoch,
                "model_state_dict": best_state,
                "best_val_auc_macro": best_auc,
                "best_val_map": va["metrics"]["map"],
                "num_classes": 14,
            }, ckpt_path)
            logger.info("  → NUEVO MEJOR: AUC_macro=%.4f mAP=%.4f",
                        best_auc, va["metrics"]["map"])
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("  Early stopping (patience=%d).", patience)
                break

    logger.info("Fase %s terminada. Mejor AUC_macro=%.4f",
                phase_name, best_auc)
    return best_state, history, best_auc


# ─── Entrenamiento principal ──────────────────────────────────────
def train_model_multilabel(model, train_loader, val_loader,
                            cfg, device, logger) -> dict:
    """
    Entrena el CNN-ViT multi-label en 2 fases.
    Misma estrategia que Sprint 3 pero con BCE loss.
    """
    ckpt_dir = Path(cfg.paths["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = cfg.training.get("use_amp", False) and torch.cuda.is_available()

    # pos_weight para BCEWithLogitsLoss
    if cfg.training.get("use_pos_weight", True):
        logger.info("Calculando pos_weight desde train set...")
        pos_weight = train_loader.dataset.compute_pos_weight(device)
        logger.info("pos_weight (min=%.2f, max=%.2f, mean=%.2f)",
                    pos_weight.min().item(),
                    pos_weight.max().item(),
                    pos_weight.mean().item())
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    p1 = cfg.training["phase1"]
    p2 = cfg.training["phase2"]
    all_history = []

    # ─── FASE 1: CNN congelada ────────────────────────────────────
    logger.info("Iniciando Fase 1: CNN congelada, ViT entrena")
    model.set_phase(1)

    opt1 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=p1["learning_rate"],
        weight_decay=p1.get("weight_decay", 1e-4),
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=p1["epochs"], eta_min=1e-6
    )

    bs1, h1, auc_p1 = run_phase_ml(
        model, train_loader, val_loader, opt1, criterion, sched1,
        epochs=p1["epochs"], patience=p1.get("patience", 5),
        device=device, use_amp=use_amp, logger=logger,
        phase_name="P1",
        ckpt_path=ckpt_dir / "sprint4ml_phase1_best.pt",
    )
    all_history.extend(h1)
    if bs1:
        model.load_state_dict(bs1)

    # ─── FASE 2: Full FT con LR diferenciado ─────────────────────
    logger.info("Iniciando Fase 2: full fine-tuning con LR diferenciado")
    model.set_phase(2)

    param_groups = model.get_param_groups(
        lr_cnn=p2["lr_cnn"],
        lr_vit=p2["lr_vit"],
    )
    opt2 = torch.optim.Adam(param_groups,
                             weight_decay=p2.get("weight_decay", 1e-4))

    sched_cfg = p2.get("scheduler", {})
    sched2 = None
    if sched_cfg.get("enabled", False):
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2,
            T_max=sched_cfg.get("T_max", p2["epochs"]),
            eta_min=sched_cfg.get("eta_min", 1e-7),
        )

    bs2, h2, auc_p2 = run_phase_ml(
        model, train_loader, val_loader, opt2, criterion, sched2,
        epochs=p2["epochs"], patience=p2.get("patience", 10),
        device=device, use_amp=use_amp, logger=logger,
        phase_name="P2",
        ckpt_path=ckpt_dir / "sprint4ml_phase2_best.pt",
        start_auc=auc_p1,
    )
    all_history.extend(h2)

    best_p2_map = max(
        h["val_metrics"]["map"] for h in h2
        if not np.isnan(h["val_metrics"]["map"])
    ) if h2 else 0.0

    logger.info("=" * 70)
    logger.info("ENTRENAMIENTO MULTI-LABEL TERMINADO")
    logger.info("  Fase 1 best AUC_macro: %.4f", auc_p1)
    logger.info("  Fase 2 best AUC_macro: %.4f", auc_p2)
    logger.info("  Fase 2 best mAP:       %.4f", best_p2_map)
    logger.info("=" * 70)

    return {
        "history":             all_history,
        "best_val_auc_phase1": float(auc_p1),
        "best_val_auc_phase2": float(auc_p2),
        "best_val_map_phase2": float(best_p2_map),
        "checkpoint_path":     str(ckpt_dir / "sprint4ml_phase2_best.pt"),
        "num_classes":         14,
    }
