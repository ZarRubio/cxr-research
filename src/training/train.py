"""
src/training/train.py  (v1.1 - fix de autocast API)

FIX v1.1: uso consistente de la nueva API torch.amp.
  Antes:  from torch.cuda.amp import GradScaler, autocast (API vieja)
          + autocast(device_type='cuda', ...) (API nueva) → choque.
  Ahora:  from torch.amp import GradScaler, autocast (API nueva en ambos).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # <-- FIX: torch.amp, no torch.cuda.amp
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics, logits_to_probs
from src.training.utils import save_checkpoint


# ---------------------------------------------------------------------------
# Una epoch de entrenamiento
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device, use_amp: bool,
                    scaler) -> dict:
    """Corre una epoch de entrenamiento, devuelve loss y metricas."""
    model.train()
    losses = []
    all_probs = []
    all_labels = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        with torch.no_grad():
            probs = logits_to_probs(logits.detach().float(), positive_class=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    mean_loss = float(np.mean(losses))

    return {
        'loss': mean_loss,
        'metrics': metrics,
    }


# ---------------------------------------------------------------------------
# Una epoch de validacion
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader,
                       criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    losses = []
    all_probs = []
    all_labels = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        losses.append(loss.item())
        probs = logits_to_probs(logits.float(), positive_class=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    mean_loss = float(np.mean(losses))

    return {
        'loss': mean_loss,
        'metrics': metrics,
    }


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, cfg: Any,
                device: torch.device, logger) -> dict:
    """Entrena el modelo con early stopping y guarda el mejor checkpoint."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("Parametros entrenables: %d tensors, %d scalars",
                len(trainable), sum(p.numel() for p in trainable))

    optimizer_name = cfg.training.get('optimizer', 'adam').lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            trainable,
            lr=cfg.training['learning_rate'],
            weight_decay=cfg.training.get('weight_decay', 0.0),
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            trainable,
            lr=cfg.training['learning_rate'],
            momentum=0.9,
            weight_decay=cfg.training.get('weight_decay', 0.0),
        )
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' no soportado")

    criterion = nn.CrossEntropyLoss()

    use_amp = cfg.training.get('use_amp', False) and torch.cuda.is_available()
    # FIX: GradScaler tambien usa la nueva API ahora
    scaler = GradScaler(device='cuda') if use_amp else None
    logger.info("Mixed precision (AMP): %s", use_amp)

    epochs = cfg.training['epochs']
    patience = cfg.training.get('patience', 3)

    ckpt_dir = Path(cfg.paths['checkpoints'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / 'baseline_best.pt'

    best_val_auc = -float('inf')
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0

    history = []

    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento: %d epochs, patience=%d", epochs, patience)
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(model, train_loader, optimizer, criterion,
                                    device, use_amp, scaler)
        val_res = validate_one_epoch(model, val_loader, criterion, device)

        tr_m = train_res['metrics']
        va_m = val_res['metrics']

        logger.info(
            "Epoch %2d/%d | train loss=%.4f AUC=%.4f acc=%.4f | "
            "val loss=%.4f AUC=%.4f acc=%.4f sens=%.4f spec=%.4f",
            epoch, epochs,
            train_res['loss'], tr_m.auc, tr_m.accuracy,
            val_res['loss'], va_m.auc, va_m.accuracy, va_m.sensitivity, va_m.specificity,
        )

        history.append({
            'epoch': epoch,
            'train_loss': train_res['loss'],
            'train_metrics': tr_m.as_dict(),
            'val_loss': val_res['loss'],
            'val_metrics': va_m.as_dict(),
        })

        current_score = va_m.auc if not np.isnan(va_m.auc) else va_m.accuracy

        if current_score > best_val_auc:
            best_val_auc = current_score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            logger.info("  -> Nuevo mejor modelo (val AUC=%.4f)", best_val_auc)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping: %d epochs sin mejora en val AUC.",
                            patience)
                break

    if best_state_dict is not None:
        save_checkpoint({
            'epoch': best_epoch,
            'model_state_dict': best_state_dict,
            'best_val_auc': best_val_auc,
            'cfg_snapshot': {
                'backbone': cfg.model['backbone'],
                'num_classes': cfg.model['num_classes'],
                'finetune_mode': cfg.model['finetune_mode'],
            },
        }, best_ckpt_path)
        logger.info("Mejor modelo guardado en: %s (epoch %d, val AUC=%.4f)",
                    best_ckpt_path, best_epoch, best_val_auc)

    logger.info("=" * 70)
    logger.info("Entrenamiento terminado.")

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'checkpoint_path': str(best_ckpt_path),
    }
