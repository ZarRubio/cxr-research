"""
scripts/run_eval_s2.py
Evaluación final del Sprint 2 sobre test set.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih_hdf5 import NIHDatasetHDF5
from src.datasets.transforms import build_eval_transform
from src.models.baseline import build_baseline_model
from src.evaluation.metrics_s2 import compute_metrics_multiclass, format_metrics
from src.training.utils import get_device, get_logger, load_config, load_checkpoint


CLASS_NAMES = {0: "No Finding", 1: "Cardiomegaly",
               2: "Effusion",   3: "Infiltration"}


def main():
    h5_path       = os.environ.get("H5_PATH",       "/content/nih_images.h5")
    processed_dir = os.environ.get("PROCESSED_DIR", "/content/processed_s2")

    cfg    = load_config(CODE_DIR / "configs" / "sprint2.yaml")
    device = get_device()

    Path(cfg.paths["logs"]).mkdir(parents=True, exist_ok=True)
    logger = get_logger("eval_s2",
                        log_file=Path(cfg.paths["logs"]) / "eval_s2.log")

    logger.info("=" * 70)
    logger.info("RUN EVAL Sprint 2 — test set")
    logger.info("=" * 70)

    # Dataset de test
    eval_tfm = build_eval_transform(image_size=cfg.preprocessing["image_size"])
    test_ds  = NIHDatasetHDF5(f"{processed_dir}/test.csv", h5_path, eval_tfm)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    logger.info("Test set: %d imágenes", len(test_ds))
    dist = test_ds.class_distribution()
    for label, name in CLASS_NAMES.items():
        logger.info("  %s: %d", name, dist.get(label, 0))

    # Cargar mejor checkpoint
    ckpt_path = Path(cfg.paths["checkpoints"]) / "sprint2_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    model = build_baseline_model(cfg).to(device)
    ckpt  = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Checkpoint cargado: epoch=%d | val AUC_macro=%.4f",
                ckpt["epoch"], ckpt["best_val_auc_macro"])

    # Inferencia
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["image"].to(device))
            probs  = torch.softmax(logits.float(), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Métricas
    metrics = compute_metrics_multiclass(all_labels, all_probs)

    logger.info("=" * 70)
    logger.info("MÉTRICAS FINALES SOBRE TEST:")
    logger.info(format_metrics(metrics))
    logger.info("=" * 70)

    # Guardar resultados
    out_path = Path(cfg.paths["experiments"]) / "test_metrics_s2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "metrics": metrics,
            "probs":   all_probs.tolist(),
            "labels":  all_labels.tolist(),
            "checkpoint_epoch": int(ckpt["epoch"]),
        }, f, indent=2)
    logger.info("Guardado en: %s", out_path)


if __name__ == "__main__":
    main()
