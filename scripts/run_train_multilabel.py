"""
scripts/run_train_multilabel.py
Entrypoint Sprint 4 multi-label — 14 clases NIH.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih_multilabel import build_nih_multilabel_datasets
from src.datasets.transforms import build_train_transform, build_eval_transform
from src.models.cnn_vit import build_cnn_vit
from src.training.train_multilabel import train_model_multilabel
from src.training.utils import get_device, get_logger, load_config, set_seed


def main():
    h5_path       = os.environ.get("H5_PATH",       "/content/nih_images.h5")
    processed_dir = os.environ.get("PROCESSED_DIR", "/content/processed_s4ml")

    cfg    = load_config(CODE_DIR / "configs" / "sprint4_multilabel.yaml")
    set_seed(cfg.project["seed"])

    Path(cfg.paths["logs"]).mkdir(parents=True, exist_ok=True)
    logger = get_logger("train_ml",
                        log_file=Path(cfg.paths["logs"]) / "train_ml.log")

    logger.info("=" * 70)
    logger.info("SPRINT 4 MULTI-LABEL — 14 clases NIH | CNN-ViT 2 fases")
    logger.info("=" * 70)

    device = get_device()
    logger.info("Device: %s%s", device,
                f" | {torch.cuda.get_device_name(0)}" if device.type=="cuda" else "")

    train_tfm = build_train_transform(cfg.preprocessing["image_size"])
    eval_tfm  = build_eval_transform(cfg.preprocessing["image_size"])

    train_ds, val_ds, test_ds = build_nih_multilabel_datasets(
        processed_dir, h5_path, train_tfm, eval_tfm
    )
    logger.info("Train: %d | Val: %d | Test: %d",
                len(train_ds), len(val_ds), len(test_ds))

    dist = train_ds.class_distribution()
    for cls, count in dist.items():
        logger.info("  %s: %d imgs positivas", cls, count)

    nw = min(cfg.training.get("num_workers", 2), 4)
    train_loader = DataLoader(train_ds, batch_size=cfg.training["batch_size"],
                              shuffle=True, num_workers=nw,
                              pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training["batch_size"],
                              shuffle=False, num_workers=nw,
                              pin_memory=(device.type=="cuda"))

    model = build_cnn_vit(cfg).to(device)
    logger.info("Modelo CNN-ViT multi-label: %d clases | %d params",
                cfg.model["num_classes"], model.total_params())

    result = train_model_multilabel(model, train_loader, val_loader,
                                     cfg, device, logger)

    hp = Path(cfg.paths["experiments"]) / "history_s4ml.json"
    hp.parent.mkdir(parents=True, exist_ok=True)
    with hp.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("History: %s", hp)
    logger.info("RESUMEN FINAL:")
    logger.info("  P1 AUC: %.4f | P2 AUC: %.4f | P2 mAP: %.4f",
                result["best_val_auc_phase1"],
                result["best_val_auc_phase2"],
                result["best_val_map_phase2"])


if __name__ == "__main__":
    main()
