"""
scripts/run_train.py

Entrypoint del entrenamiento baseline.

Hace:
  1. Carga config.
  2. Fija seed.
  3. Construye datasets, loaders, modelo.
  4. Llama a train_model() del src/training/train.py.
  5. Guarda history.json para poder plottear curvas despues.

Uso (desde Colab):
    !python /content/drive/MyDrive/Tesis_CXR/code/scripts/run_train.py
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

from src.datasets.nih import build_nih_datasets
from src.datasets.transforms import build_train_transform, build_eval_transform
from src.models.baseline import build_baseline_model
from src.training.train import train_model
from src.training.utils import (
    get_device,
    get_logger,
    load_config,
    set_seed,
)


def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = CODE_DIR / "configs" / "baseline.yaml"
    cfg = load_config(config_path)

    set_seed(cfg.project['seed'])

    log_file = Path(cfg.paths['logs']) / "train.log"
    logger = get_logger("train", log_file=log_file)

    logger.info("=" * 70)
    logger.info("RUN TRAIN - Sprint 1 baseline")
    logger.info("Config: %s", config_path)
    logger.info("=" * 70)

    device = get_device()
    logger.info("Device: %s", device)
    if device.type == 'cuda':
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # 1. Datasets y loaders
    train_tfm = build_train_transform(image_size=cfg.preprocessing['image_size'])
    eval_tfm = build_eval_transform(image_size=cfg.preprocessing['image_size'])

    train_ds, val_ds, test_ds = build_nih_datasets(cfg, train_tfm, eval_tfm)
    logger.info("Train: %d | Val: %d | Test: %d",
                len(train_ds), len(val_ds), len(test_ds))

    # Para train queremos shuffle; para val/test no (reproducibilidad).
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

    # 2. Modelo
    model = build_baseline_model(cfg).to(device)
    logger.info("Modelo: %s (backbone=%s, mode=%s)",
                model.__class__.__name__,
                cfg.model['backbone'], cfg.model['finetune_mode'])
    logger.info("  total params: %d | trainable: %d",
                model.total_params(), model.trainable_params())

    # 3. Entrenamiento
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        logger=logger,
    )

    # 4. Guardar history.json
    history_path = Path(cfg.paths['experiments']) / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open('w', encoding='utf-8') as f:
        json.dump({
            'history': result['history'],
            'best_epoch': result['best_epoch'],
            'best_val_auc': result['best_val_auc'],
            'checkpoint_path': result['checkpoint_path'],
        }, f, indent=2, ensure_ascii=False)

    logger.info("History guardado en: %s", history_path)
    logger.info("RESUMEN: best epoch=%d | best val AUC=%.4f",
                result['best_epoch'], result['best_val_auc'])


if __name__ == "__main__":
    main()
