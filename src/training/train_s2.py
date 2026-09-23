"""
src/training/train_s2.py

Loop de entrenamiento para Sprint 2: 4 clases, finetune full, LR scheduler,
class weights.

Diferencias respecto a Sprint 1 (train.py):
  - CrossEntropyLoss con class_weights para compensar desbalance residual.
  - ReduceLROnPlateau scheduler: reduce LR si val_loss no mejora.
  - Logging de AUC macro + AUC por clase (no solo AUC binario).
  - Compatibilidad con num_classes configurable (2, 3 o 4).
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

from src.evaluation.metrics_s2 import compute_metrics_multiclass, logits_to_probs_multiclass
from src.training.utils import save_checkpoint


# ---------------------------------------------------------------------------
# Una epoch de entrenamiento
# ---------------------------------------------------------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    use_amp: bool,
                    scaler: GradScaler | None) -> dict:
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
            probs = logits_to_probs_multiclass(logits.detach().float())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics_multiclass(all_labels, all_probs)
    return {'loss': float(np.mean(losses)), 'metrics': metrics}


# ---------------------------------------------------------------------------
# Una epoch de validación
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader,
                       criterion: nn.Module,
                       device: torch.device) -> dict:
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
        probs = logits_to_probs_multiclass(logits.float())
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics_multiclass(all_labels, all_probs)
    return {'loss': float(np.mean(losses)), 'metrics': metrics}


# ---------------------------------------------------------------------------
# Calcular class weights desde el dataset de entrenamiento
# ---------------------------------------------------------------------------
def compute_class_weights(train_loader: DataLoader,
                          num_classes: int,
                          device: torch.device) -> torch.Tensor:
    """
    Calcula pesos inversamente proporcionales a la frecuencia de cada clase.
    Peso_clase = n_total / (n_clases * n_clase_i)
    """
    counts = torch.zeros(num_classes)
    for batch in train_loader:
        for label in batch['label']:
            counts[label.item()] += 1

    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.sum() * num_classes  # normalizar
    return weights.to(device)


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------
def train_model_s2(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   cfg: Any,
                   device: torch.device,
                   logger) -> dict:
    """
    Entrena el modelo Sprint 2 con:
      - CrossEntropyLoss + class weights opcionales.
      - ReduceLROnPlateau scheduler.
      - Early stopping por val AUC macro.
      - Checkpointing del mejor modelo.
    """
    num_classes = cfg.model['num_classes']

    # Parámetros entrenables
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("Parámetros entrenables: %d tensors, %d scalars",
                len(trainable), sum(p.numel() for p in trainable))

    # Optimizer
    lr = cfg.training['learning_rate']
    wd = cfg.training.get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    logger.info("Optimizer: Adam | LR=%.2e | WD=%.2e", lr, wd)

    # Class weights
    use_weights = cfg.training.get('use_class_weights', False)
    if use_weights:
        logger.info("Calculando class weights...")
        weights = compute_class_weights(train_loader, num_classes, device)
        logger.info("Class weights: %s", weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Sin class weights.")

    # LR Scheduler
    scheduler_cfg = cfg.training.get('scheduler', {})
    use_scheduler = scheduler_cfg.get('enabled', False)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get('mode', 'min'),
            factor=scheduler_cfg.get('factor', 0.5),
            patience=scheduler_cfg.get('patience', 2),
            min_lr=scheduler_cfg.get('min_lr', 1e-6),
        )
        logger.info("Scheduler: ReduceLROnPlateau | factor=%.1f | patience=%d",
                    scheduler_cfg.get('factor', 0.5),
                    scheduler_cfg.get('patience', 2))

    # AMP
    use_amp = cfg.training.get('use_amp', False) and torch.cuda.is_available()
    scaler = GradScaler(device='cuda') if use_amp else None
    logger.info("Mixed precision (AMP): %s", use_amp)

    epochs = cfg.training['epochs']
    patience = cfg.training.get('patience', 5)

    ckpt_dir = Path(cfg.paths['checkpoints'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / 'sprint2_best.pt'

    best_val_auc = -float('inf')
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0
    history = []

    logger.info("=" * 70)
    logger.info("Iniciando entrenamiento S2: %d epochs | patience=%d | "
                "num_classes=%d | finetune=%s",
                epochs, patience, num_classes, cfg.model['finetune_mode'])
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']

        train_res = train_one_epoch(model, train_loader, optimizer,
                                    criterion, device, use_amp, scaler)
        val_res = validate_one_epoch(model, val_loader, criterion, device)

        tr_m = train_res['metrics']
        va_m = val_res['metrics']

        # Step del scheduler sobre val_loss
        if scheduler is not None:
            scheduler.step(val_res['loss'])

        logger.info(
            "Epoch %2d/%d | LR=%.2e | "
            "train loss=%.4f AUC_macro=%.4f | "
            "val loss=%.4f AUC_macro=%.4f acc=%.4f",
            epoch, epochs, current_lr,
            train_res['loss'], tr_m['auc_macro'],
            val_res['loss'], va_m['auc_macro'], va_m['accuracy'],
        )

        # Log AUC por clase en val
        for i, auc in enumerate(va_m['auc_per_class']):
            logger.info("  val AUC clase %d: %.4f", i, auc)

        history.append({
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_res['loss'],
            'train_metrics': tr_m,
            'val_loss': val_res['loss'],
            'val_metrics': va_m,
        })

        # Early stopping por AUC macro
        score = va_m['auc_macro']
        if score > best_val_auc:
            best_val_auc = score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            logger.info("  → Nuevo mejor modelo (val AUC_macro=%.4f)", best_val_auc)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping: %d epochs sin mejora.", patience)
                break

    # Guardar mejor checkpoint
    if best_state_dict is not None:
        save_checkpoint({
            'epoch': best_epoch,
            'model_state_dict': best_state_dict,
            'best_val_auc_macro': best_val_auc,
            'num_classes': num_classes,
            'cfg_snapshot': {
                'backbone': cfg.model['backbone'],
                'num_classes': cfg.model['num_classes'],
                'finetune_mode': cfg.model['finetune_mode'],
            },
        }, best_ckpt_path)
        logger.info("Mejor modelo guardado: %s (epoch %d, AUC_macro=%.4f)",
                    best_ckpt_path, best_epoch, best_val_auc)

    logger.info("=" * 70)
    logger.info("Entrenamiento S2 terminado.")

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_auc_macro': best_val_auc,
        'checkpoint_path': str(best_ckpt_path),
    }
