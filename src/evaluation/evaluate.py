"""
src/evaluation/evaluate.py

Evaluacion final del modelo sobre el test set.

Carga el checkpoint guardado durante entrenamiento, corre inferencia sobre test,
y calcula las 5 metricas del charter + curva ROC + matriz de confusion.

Los resultados se guardan en experiments/:
  - test_metrics.json: las 5 metricas + probabilidades + labels
  - test_roc.png: curva ROC
  - test_confusion_matrix.png: matriz de confusion visualizada
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics, logits_to_probs


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, test_loader: DataLoader,
                   device: torch.device, threshold: float = 0.5,
                   logger=None) -> dict:
    """
    Corre inferencia sobre test_loader y devuelve metricas + raw predictions.

    Returns:
        {
          'metrics': ClassificationMetrics,
          'probs': np.ndarray (N,) probabilidades de clase positiva,
          'labels': np.ndarray (N,) labels verdaderos,
          'image_indices': list[str] nombres de archivos correspondientes,
        }
    """
    model.eval()

    all_probs = []
    all_labels = []
    all_names = []

    for batch in test_loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label']
        names = batch['image_index']

        logits = model(images)
        probs = logits_to_probs(logits.float(), positive_class=1)

        all_probs.append(probs.cpu())
        all_labels.append(labels)
        all_names.extend(names)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)

    if logger is not None:
        logger.info("Evaluacion sobre test: %d muestras", len(all_labels))
        logger.info("  %s", metrics)

    return {
        'metrics': metrics,
        'probs': all_probs,
        'labels': all_labels,
        'image_indices': all_names,
    }


def save_test_results(result: dict, output_dir: Path | str) -> None:
    """Guarda test_metrics.json con los resultados de la evaluacion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        'metrics': result['metrics'].as_dict(),
        'probs': result['probs'].tolist(),
        'labels': result['labels'].tolist(),
        'image_indices': list(result['image_indices']),
    }

    path = output_dir / 'test_metrics.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_model_from_checkpoint(checkpoint_path: Path | str, cfg,
                               device: torch.device) -> torch.nn.Module:
    """
    Reconstruye el modelo y carga los pesos del checkpoint.

    Importante: necesita la misma config que se uso en entrenamiento.
    """
    from src.models.baseline import build_baseline_model

    model = build_baseline_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
