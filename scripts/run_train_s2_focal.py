"""
scripts/run_train_s2_focal.py
Entrypoint Sprint 2 con Focal Loss.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih_hdf5 import build_nih_hdf5_datasets
from src.datasets.transforms import build_train_transform, build_eval_transform
from src.models.baseline import build_baseline_model
from src.training.train_s2_fl import train_model_focal
from src.training.utils import get_device, get_logger, load_config, set_seed


def main():
    h5_path       = os.environ.get("H5_PATH",       "/content/nih_images.h5")
    processed_dir = os.environ.get("PROCESSED_DIR", "/content/processed_s2")
    cfg    = load_config(CODE_DIR / "configs" / "sprint2.yaml")
    set_seed(cfg.project["seed"])
    Path(cfg.paths["logs"]).mkdir(parents=True, exist_ok=True)
    logger = get_logger("train_s2_focal",
                        log_file=Path(cfg.paths["logs"]) / "train_s2_focal.log")
    logger.info("=" * 70)
    logger.info("RUN TRAIN S2 — FOCAL LOSS | 4 clases | finetune full")
    logger.info("=" * 70)
    device = get_device()
    logger.info("Device: %s%s", device,
                f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else "")
    train_tfm = build_train_transform(cfg.preprocessing["image_size"])
    eval_tfm  = build_eval_transform(cfg.preprocessing["image_size"])
    train_ds, val_ds, _ = build_nih_hdf5_datasets(
        processed_dir, h5_path, train_tfm, eval_tfm)
    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))
    nw = min(cfg.training.get("num_workers", 2), 4)
    train_loader = DataLoader(train_ds, batch_size=cfg.training["batch_size"],
                              shuffle=True, num_workers=nw,
                              pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training["batch_size"],
                              shuffle=False, num_workers=nw,
                              pin_memory=(device.type=="cuda"))
    model = build_baseline_model(cfg).to(device)
    logger.info("Modelo: %s | mode=%s | trainable=%d",
                model.__class__.__name__, cfg.model["finetune_mode"],
                model.trainable_params())
    result = train_model_focal(model, train_loader, val_loader, cfg, device, logger)
    hp = Path(cfg.paths["experiments"]) / "history_s2_focal.json"
    hp.parent.mkdir(parents=True, exist_ok=True)
    with hp.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("History: %s", hp)
    logger.info("RESUMEN: best epoch=%d | AUC_macro=%.4f",
                result["best_epoch"], result["best_val_auc_macro"])


if __name__ == "__main__":
    main()
