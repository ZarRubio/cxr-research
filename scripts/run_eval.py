"""
scripts/run_eval.py

Entrypoint: evalua el mejor checkpoint sobre test set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.datasets.nih import build_nih_datasets
from src.datasets.transforms import build_train_transform, build_eval_transform
from src.evaluation.evaluate import (
    evaluate_model,
    load_model_from_checkpoint,
    save_test_results,
)
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

    log_file = Path(cfg.paths['logs']) / "evaluate.log"
    logger = get_logger("evaluate", log_file=log_file)

    logger.info("=" * 70)
    logger.info("RUN EVAL - Sprint 1 baseline")
    logger.info("=" * 70)

    device = get_device()
    logger.info("Device: %s", device)

    # 1. Dataset de test (con transform de evaluacion, SIN augmentation)
    train_tfm = build_train_transform(image_size=cfg.preprocessing['image_size'])
    eval_tfm = build_eval_transform(image_size=cfg.preprocessing['image_size'])
    _, _, test_ds = build_nih_datasets(cfg, train_tfm, eval_tfm)

    test_loader = DataLoader(
        test_ds, batch_size=cfg.training['batch_size'], shuffle=False,
        num_workers=cfg.training.get('num_workers', 0),
        pin_memory=(device.type == 'cuda'),
    )
    logger.info("Test: %d muestras | distribucion: %s",
                len(test_ds), test_ds.class_distribution())

    # 2. Cargar checkpoint
    ckpt_path = Path(cfg.paths['checkpoints']) / 'baseline_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {ckpt_path}. "
            "Corre primero scripts/run_train.py."
        )
    logger.info("Checkpoint: %s", ckpt_path)
    model = load_model_from_checkpoint(ckpt_path, cfg, device)

    # 3. Evaluar
    threshold = cfg.evaluation.get('threshold', 0.5)
    result = evaluate_model(model, test_loader, device,
                            threshold=threshold, logger=logger)

    # 4. Guardar
    save_test_results(result, Path(cfg.paths['experiments']))
    logger.info("Resultados guardados en: %s/test_metrics.json",
                cfg.paths['experiments'])

    logger.info("=" * 70)
    logger.info("RESUMEN TEST:")
    logger.info("  %s", result['metrics'])


if __name__ == "__main__":
    main()
