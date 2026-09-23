"""
src/training/utils.py
Utilidades transversales: seeding, logging, device, checkpoints, config.
Todos los demas modulos (datasets, models, training, evaluation) importan de aqui.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Fija todas las fuentes de aleatoriedad relevantes.

    Nota: deterministic=True puede reducir ~5-15% el throughput de entrenamiento,
    pero es obligatorio para un baseline de tesis. No lo desactives sin justificarlo.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device(prefer: str = "auto") -> torch.device:
    """Devuelve el device a usar. prefer: 'auto', 'cuda', 'mps' o 'cpu'."""
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "cxr", log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Crea un logger con formato consistente.
    Si se pasa log_file, tambien escribe a disco (util para cada experimento).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Contenedor simple para acceder a la config como atributos."""
    data: dict

    def __getattr__(self, item: str) -> Any:
        try:
            return self.data[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def load_config(path: str | Path) -> Config:
    """Carga un YAML a un objeto Config accesible por atributos."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro la config: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(cfg)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, path: str | Path) -> None:
    """Guarda un checkpoint (model_state, optimizer_state, epoch, metrics, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Carga un checkpoint previamente guardado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el checkpoint: {path}")
    return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# Sanity check manual
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)
    logger = get_logger("cxr")
    logger.info("Device detectado: %s", get_device())
    logger.info("Utils cargado correctamente.")
