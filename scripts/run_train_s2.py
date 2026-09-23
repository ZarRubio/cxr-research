"""
scripts/run_train_s2.py

Entrypoint del entrenamiento Sprint 2 (4 clases, finetune full).

Uso (desde Colab):
    !python /content/drive/MyDrive/Tesis_CXR/code/scripts/run_train_s2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih import build_nih_datasets_from_processed
from src.datasets.transforms import build_train_transform, build_eval_transform
from src.models.baseline import build_baseline_model
from src.training.train_s2 import train_model_s2
from src.training.utils import (
    get_device, get_logger, load_config, set_seed,
)


def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = CODE_DIR / "configs" / "sprint2.yaml"
    cfg = load_config(config_path)

    set_seed(cfg.project['seed'])

    log_file = Path(cfg.paths['logs']) / "train_s2.log"
    logger = get_logger("train_s2", log_file=log_file)

    logger.info("=" * 70)
    logger.info("RUN TRAIN Sprint 2 — 4 clases, finetune full")
    logger.info("Config: %s", config_path)
    logger.info("=" * 70)

    device = get_device()
    logger.info("Device: %s", device)
    if device.type == 'cuda':
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Datasets y loaders
    train_tfm = build_train_transform(image_size=cfg.preprocessing['image_size'])
    eval_tfm  = build_eval_transform(image_size=cfg.preprocessing['image_size'])

    # Los splits del Sprint 2 están en data/processed_s2/
    train_ds, val_ds, test_ds = build_nih_datasets_from_processed(
        processed_dir=cfg.paths['data_processed'],
        images_dir=str(Path(cfg.paths['data_raw']) / cfg.dataset['images_subdir']),
        train_transform=train_tfm,
        eval_transform=eval_tfm,
    )
    logger.info("Train: %d | Val: %d | Test: %d",
                len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training['batch_size'], shuffle=True,
        num_workers=cfg.training.get('num_workers', 0),
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training['batch_size'], shuffle=False,
        num_workers=cfg.training.get('num_workers', 0),
        pin_memory=(device.type == 'cuda'),
    )

    # Modelo (4 clases, finetune full)
    model = build_baseline_model(cfg).to(device)
    logger.info("Modelo: %s | backbone=%s | mode=%s | num_classes=%d",
                model.__class__.__name__,
                cfg.model['backbone'],
                cfg.model['finetune_mode'],
                cfg.model['num_classes'])
    logger.info("  total params: %d | trainable: %d",
                model.total_params(), model.trainable_params())

    # Entrenamiento
    result = train_model_s2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        logger=logger,
    )

    # Guardar history
    history_path = Path(cfg.paths['experiments']) / "history_s2.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("History guardado: %s", history_path)
    logger.info("RESUMEN: best epoch=%d | best val AUC_macro=%.4f",
                result['best_epoch'], result['best_val_auc_macro'])


if __name__ == "__main__":
    main()
