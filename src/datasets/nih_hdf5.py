"""
src/datasets/nih_hdf5.py

Dataset de PyTorch que lee imágenes desde un archivo HDF5.
Reemplaza nih.py para entrenamiento en Colab sin errores de Drive I/O.

Ventajas sobre leer PNGs sueltos desde Drive:
  - Un solo archivo → copia rápida a /content/ al inicio de sesión.
  - Lectura 10-20x más rápida (acceso secuencial vs random a 25k archivos).
  - Compatible con num_workers > 0 (h5py soporta acceso paralelo).
  - Sin OSError Errno 5 de Drive.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NIHDatasetHDF5(Dataset):
    """
    Dataset NIH que lee imágenes desde un archivo HDF5.

    El HDF5 tiene la estructura:
      /image_names: array de strings con nombres de archivo
      /images/{nombre.png}: array uint8 (H, W) en escala de grises

    Args:
        csv_path: CSV con columnas Image Index, label, Patient ID, etc.
        h5_path: ruta al archivo nih_images.h5.
        transform: pipeline de preprocesamiento (de transforms.py).
    """

    def __init__(self, csv_path: str | Path, h5_path: str | Path,
                 transform=None):
        self.csv_path = Path(csv_path)
        self.h5_path  = Path(h5_path)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {self.csv_path}")
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"HDF5 no encontrado: {self.h5_path}\n"
                "Corre primero S2_05_create_hdf5.ipynb"
            )

        self.df = pd.read_csv(self.csv_path)
        self._validate_columns()

        # Abrir HDF5 en modo lectura (se mantiene abierto durante el entrenamiento)
        # NOTA: h5py con swmr=False es thread-safe para lectura.
        self._h5 = h5py.File(self.h5_path, 'r')

    def _validate_columns(self) -> None:
        required = {'Image Index', 'label', 'Patient ID'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        label = int(row['label'])

        # Leer desde HDF5 (mucho más rápido que abrir PNG desde Drive)
        arr = self._h5['images'][image_name][:]  # (H, W) uint8

        if self.transform is not None:
            arr = self.transform(arr)

        return {
            'image': arr,
            'label': torch.tensor(label, dtype=torch.long),
            'image_index': image_name,
        }

    def class_distribution(self) -> dict:
        counts = self.df['label'].value_counts().to_dict()
        return {int(k): int(v) for k, v in counts.items()}

    def __del__(self):
        # Cerrar el archivo HDF5 al destruir el objeto
        if hasattr(self, '_h5') and self._h5.id.valid:
            self._h5.close()


def build_nih_hdf5_datasets(processed_dir: str, h5_path: str,
                             train_transform=None,
                             eval_transform=None) -> tuple:
    """Crea los 3 datasets (train/val/test) leyendo desde HDF5."""
    processed = Path(processed_dir)
    h5 = Path(h5_path)
    train_ds = NIHDatasetHDF5(processed / "train.csv", h5, train_transform)
    val_ds   = NIHDatasetHDF5(processed / "val.csv",   h5, eval_transform)
    test_ds  = NIHDatasetHDF5(processed / "test.csv",  h5, eval_transform)
    return train_ds, val_ds, test_ds
