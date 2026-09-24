"""Evaluate one trained 14-label model and write aggregate-only results.

Thresholds are selected on validation data. Test metrics and confidence
intervals are then computed without writing per-image predictions or IDs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih_multilabel import CLASSES_14, NIHDatasetMultiLabel
from src.datasets.transforms import build_eval_transform
from src.datasets.transforms_multilabel_v2 import build_eval_transform_v2
from src.models.cnn_vit import build_cnn_vit
from src.training.utils import get_device, load_config, set_seed


def predict(model, dataset, batch_size: int, num_workers: int,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def thresholds_from_validation(y_true: np.ndarray,
                               y_probs: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    for c in range(y_true.shape[1]):
        if np.unique(y_true[:, c]).size < 2:
            continue
        fpr, tpr, values = roc_curve(y_true[:, c], y_probs[:, c])
        candidate = values[int(np.argmax(tpr - fpr))]
        if np.isfinite(candidate):
            thresholds[c] = candidate
    return thresholds


def evaluate(y_true: np.ndarray, y_probs: np.ndarray,
             thresholds: np.ndarray) -> dict:
    per_class = {}
    aucs, aps, f1s, precisions, sensitivities, specificities = [], [], [], [], [], []
    for c, name in enumerate(CLASSES_14):
        truth = y_true[:, c].astype(bool)
        pred = y_probs[:, c] >= thresholds[c]
        tp = int(np.logical_and(pred, truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        tn = int(np.logical_and(~pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if precision + sensitivity else 0.0)
        if np.unique(y_true[:, c]).size == 2:
            auc = float(roc_auc_score(y_true[:, c], y_probs[:, c]))
            ap = float(average_precision_score(y_true[:, c], y_probs[:, c]))
            aucs.append(auc)
            aps.append(ap)
        else:
            auc = ap = None
        per_class[name] = {
            "auc": auc,
            "average_precision": ap,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "threshold": float(thresholds[c]),
            "n_positive": int(truth.sum()),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        precisions.append(precision)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1s.append(f1)
    return {
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "macro_average_precision": float(np.mean(aps)) if aps else None,
        "macro_precision": float(np.mean(precisions)),
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
    }


def patient_bootstrap_ci(y_true: np.ndarray, y_probs: np.ndarray,
                         patient_ids: np.ndarray, thresholds: np.ndarray,
                         n_bootstrap: int, seed: int) -> dict:
    """Cluster bootstrap: resample patients, retaining all their images."""
    unique_patients = np.unique(patient_ids)
    patient_rows = {
        patient: np.flatnonzero(patient_ids == patient)
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    metric_names = ("macro_auc", "macro_average_precision", "macro_f1")
    samples = {name: [] for name in metric_names}
    for _ in range(n_bootstrap):
        draw = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rows = np.concatenate([patient_rows[patient] for patient in draw])
        result = evaluate(y_true[rows], y_probs[rows], thresholds)
        for name in metric_names:
            value = result[name]
            if value is not None and np.isfinite(value):
                samples[name].append(value)
    return {
        name: {
            "lower_95": float(np.percentile(values, 2.5)) if values else None,
            "upper_95": float(np.percentile(values, 97.5)) if values else None,
            "valid_bootstraps": len(values),
        }
        for name, values in samples.items()
    }


def main() -> None:
    config_path = Path(os.environ.get(
        "CONFIG_PATH", CODE_DIR / "configs" / "sprint4_multilabel_v3.yaml"
    ))
    cfg = load_config(config_path)
    set_seed(cfg.project["seed"])
    device = get_device()
    h5_path = Path(os.environ.get(
        "H5_PATH", Path(cfg.paths["data_raw"]) / "nih_images.h5"
    ))
    processed_dir = Path(os.environ.get("PROCESSED_DIR", cfg.paths["data_processed"]))
    transform_kind = cfg.preprocessing.get("transform", "standard")
    eval_transform = (build_eval_transform_v2 if transform_kind == "multilabel_v2"
                      else build_eval_transform)
    transform = eval_transform(cfg.preprocessing["image_size"])

    val_ds = NIHDatasetMultiLabel(processed_dir / "val.csv", h5_path, transform)
    test_ds = NIHDatasetMultiLabel(processed_dir / "test.csv", h5_path, transform)
    batch_size = int(cfg.training["batch_size"])
    num_workers = min(int(cfg.training.get("num_workers", 0)), 4)
    model = build_cnn_vit(cfg).to(device)
    checkpoint_path = Path(os.environ.get(
        "CHECKPOINT_PATH",
        Path(cfg.paths["checkpoints"]) / "sprint4ml_phase2_best.pt",
    ))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_probs, val_labels = predict(model, val_ds, batch_size, num_workers, device)
    test_probs, test_labels = predict(model, test_ds, batch_size, num_workers, device)
    thresholds = thresholds_from_validation(val_labels, val_probs)
    metrics = evaluate(test_labels, test_probs, thresholds)
    bootstrap_n = int(os.environ.get("BOOTSTRAP_SAMPLES", "500"))
    intervals = patient_bootstrap_ci(
        test_labels, test_probs, test_ds.df["Patient ID"].to_numpy(),
        thresholds, bootstrap_n, seed=int(cfg.project["seed"]),
    ) if bootstrap_n > 0 else {}

    report = {
        "dataset": "NIH ChestX-ray14",
        "model": "CNN-ViT multilabel",
        "backbone_initialization": cfg.model["backbone"],
        "n_test_images": len(test_ds),
        "n_test_patients": int(test_ds.df["Patient ID"].nunique()),
        "n_validation_images": len(val_ds),
        "n_validation_patients": int(val_ds.df["Patient ID"].nunique()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "thresholds_calibrated_on": "validation",
        "macro_metrics": metrics,
        "patient_cluster_bootstrap_95ci": intervals,
        "bootstrap_samples": bootstrap_n,
        "seed": int(cfg.project["seed"]),
        "config_file": config_path.name,
        "note": "Aggregate-only report; no per-image predictions or identifiers are stored.",
    }
    report_path = Path(os.environ.get(
        "REPORT_PATH", Path(cfg.paths["experiments"]) / "test_metrics_aggregate.json"
    ))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "test_images": report["n_test_images"],
        "test_patients": report["n_test_patients"],
        "macro_auc": metrics["macro_auc"],
        "macro_average_precision": metrics["macro_average_precision"],
        "macro_f1": metrics["macro_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
