"""
scripts/run_train_s2_hdf5.py

Entrypoint Sprint 2 usando HDF5 para lectura rápida de imágenes.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih_hdf5 import build_nih_hdf5_datasets
from src.datasets.transforms import build_train_transform, build_eval_transform
from src.models.baseline import build_baseline_model
from src.training.train_s2 import train_model_s2
from src.training.utils import get_device, get_logger, load_config, set_seed


def main(config_path=None, h5_path=None, processed_dir=None):
    if config_path is None:
        config_path = CODE_DIR / "configs" / "sprint2.yaml"
    cfg = load_config(config_path)
    set_seed(cfg.project['seed'])

    log_file = Path(cfg.paths['logs']) / "train_s2.log"
    logger = get_logger("train_s2", log_file=log_file)

    logger.info("=" * 70)
    logger.info("RUN TRAIN S2 (HDF5) — 4 clases, finetune full")
    logger.info("=" * 70)

    device = get_device()
    logger.info("Device: %s | GPU: %s", device,
                torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')

    # Rutas (pueden venir de argumentos o del config)
    h5  = h5_path      or '/content/nih_images.h5'
    prc = processed_dir or cfg.paths['data_processed']

    train_tfm = build_train_transform(image_size=cfg.preprocessing['image_size'])
    eval_tfm  = build_eval_transform(image_size=cfg.preprocessing['image_size'])

    train_ds, val_ds, test_ds = build_nih_hdf5_datasets(prc, h5, train_tfm, eval_tfm)
    logger.info("Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # num_workers=2 funciona bien con HDF5 (a diferencia de Drive directo)
    nw = cfg.training.get('num_workers', 2)
    train_loader = DataLoader(train_ds, batch_size=cfg.training['batch_size'],
                              shuffle=True, num_workers=nw,
                              pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds, batch_size=cfg.training['batch_size'],
                              shuffle=False, num_workers=nw,
                              pin_memory=(device.type=='cuda'))

    model = build_baseline_model(cfg).to(device)
    logger.info("Modelo: %s | mode=%s | num_classes=%d | trainable=%d",
                model.__class__.__name__, cfg.model['finetune_mode'],
                cfg.model['num_classes'], model.trainable_params())

    result = train_model_s2(model, train_loader, val_loader, cfg, device, logger)

    history_path = Path(cfg.paths['experiments']) / "history_s2.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open('w') as f:
        json.dump(result, f, indent=2)

    logger.info("RESUMEN: best epoch=%d | best val AUC_macro=%.4f",
                result['best_epoch'], result['best_val_auc_macro'])


if __name__ == "__main__":
    main()
