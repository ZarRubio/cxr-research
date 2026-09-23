"""
scripts/run_train_s4.py
Entrypoint Sprint 4: CNN-ViT híbrido, 2 fases.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih_hdf5 import NIHDatasetHDF5
from src.datasets.transforms import build_train_transform, build_eval_transform
from src.models.cnn_vit import build_cnn_vit
from src.training.train_s4 import train_model_cnn_vit
from src.training.utils import get_device, get_logger, load_config, set_seed


def main():
    h5_path       = os.environ.get("H5_PATH",       "/content/nih_images.h5")
    processed_dir = os.environ.get("PROCESSED_DIR", "/content/processed_s4")
    cfg    = load_config(CODE_DIR / "configs" / "sprint4.yaml")
    set_seed(cfg.project["seed"])
    Path(cfg.paths["logs"]).mkdir(parents=True, exist_ok=True)
    logger = get_logger("train_s4",
                        log_file=Path(cfg.paths["logs"]) / "train_s4.log")
    logger.info("=" * 70)
    logger.info("RUN TRAIN Sprint 4 — CNN-ViT Híbrido | 2 fases")
    logger.info("  h5:        %s", h5_path)
    logger.info("  processed: %s", processed_dir)
    logger.info("=" * 70)
    device = get_device()
    logger.info("Device: %s%s", device,
                f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else "")

    train_tfm = build_train_transform(cfg.preprocessing["image_size"])
    eval_tfm  = build_eval_transform(cfg.preprocessing["image_size"])

    train_ds = NIHDatasetHDF5(f"{processed_dir}/train.csv", h5_path, train_tfm)
    val_ds   = NIHDatasetHDF5(f"{processed_dir}/val.csv",   h5_path, eval_tfm)
    test_ds  = NIHDatasetHDF5(f"{processed_dir}/test.csv",  h5_path, eval_tfm)
    logger.info("Train: %d | Val: %d | Test: %d",
                len(train_ds), len(val_ds), len(test_ds))
    logger.info("Distribución train: %s", train_ds.class_distribution())

    nw = min(cfg.training.get("num_workers", 2), 4)
    train_loader = DataLoader(train_ds, batch_size=cfg.training["batch_size"],
                              shuffle=True, num_workers=nw,
                              pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training["batch_size"],
                              shuffle=False, num_workers=nw,
                              pin_memory=(device.type=="cuda"))

    model = build_cnn_vit(cfg).to(device)
    summary = model.param_summary()
    logger.info("Modelo CNN-ViT:")
    logger.info("  CNN params:  %d", summary["cnn_total"])
    logger.info("  ViT params:  %d", summary["vit_total"])
    logger.info("  Total:       %d", summary["total"])

    result = train_model_cnn_vit(model, train_loader, val_loader,
                                  cfg, device, logger)

    hp = Path(cfg.paths["experiments"]) / "history_s4.json"
    hp.parent.mkdir(parents=True, exist_ok=True)
    with hp.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("History: %s", hp)
    logger.info("RESUMEN FINAL:")
    logger.info("  Fase 1 best AUC: %.4f", result["best_val_auc_phase1"])
    logger.info("  Fase 2 best AUC: %.4f", result["best_val_auc_phase2"])


if __name__ == "__main__":
    main()
