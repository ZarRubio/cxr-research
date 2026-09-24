"""
src/datasets/nih_multilabel.py

Dataset PyTorch para clasificación multi-label de 14 patologías NIH.

Diferencia clave respecto a nih_hdf5.py (multi-clase):
  - Retorna un vector binario de 14 posiciones en vez de un entero.
  - "No Finding" → vector de ceros [0,0,...,0].
  - Una imagen puede tener múltiples patologías activas.

Ejemplo:
  "Cardiomegaly|Effusion" → [0,1,0,0,1,0,0,0,0,0,0,0,0,0]
  "No Finding"             → [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  "Infiltration"           → [0,0,0,0,0,0,0,0,1,0,0,0,0,0]
"""
from __future__ import annotations

from pathlib import Path
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Las 14 patologías del NIH en orden canónico
CLASSES_14 = [
    "Atelectasis",        # 0
    "Cardiomegaly",       # 1
    "Consolidation",      # 2
    "Edema",              # 3
    "Effusion",           # 4
    "Emphysema",          # 5
    "Fibrosis",           # 6
    "Hernia",             # 7
    "Infiltration",       # 8
    "Mass",               # 9
    "Nodule",             # 10
    "Pleural_Thickening", # 11
    "Pneumonia",          # 12
    "Pneumothorax",       # 13
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES_14)}
NUM_CLASSES  = 14


def findings_to_vector(finding_labels: str) -> np.ndarray:
    """
    Convierte el string de hallazgos del CSV a vector binario.

    Args:
        finding_labels: "Cardiomegaly|Effusion" o "No Finding"

    Returns:
        np.ndarray de shape (14,) con 0s y 1s.
    """
    vector = np.zeros(NUM_CLASSES, dtype=np.float32)

    if finding_labels.strip() == "No Finding":
        return vector  # todo ceros

    for finding in finding_labels.split("|"):
        finding = finding.strip()
        if finding in CLASS_TO_IDX:
            vector[CLASS_TO_IDX[finding]] = 1.0

    return vector


class NIHDatasetMultiLabel(Dataset):
    """
    Dataset multi-label que lee imágenes desde HDF5 y retorna
    vectores binarios de 14 posiciones como labels.

    Args:
        csv_path:  CSV con columnas Image Index, Finding Labels, Patient ID.
        h5_path:   Ruta al archivo nih_images.h5.
        transform: Pipeline de preprocesamiento.
    """

    def __init__(self, csv_path: str | Path,
                 h5_path:  str | Path,
                 transform=None):
        self.csv_path  = Path(csv_path)
        self.h5_path   = Path(h5_path)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {self.csv_path}")
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 no encontrado: {self.h5_path}")

        self.df = pd.read_csv(self.csv_path)
        self._validate()

        # Pre-computar vectores de labels para eficiencia
        self._labels = np.stack(
            [findings_to_vector(f) for f in self.df["Finding Labels"]],
            axis=0,
        )  # (N, 14) float32

        # Open the HDF5 handle lazily per process. DataLoader workers must not
        # reuse a handle opened before fork.
        self._h5 = None
        self._h5_pid = None

    def _get_h5(self):
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid or not self._h5.id.valid:
            if self._h5 is not None and self._h5.id.valid:
                self._h5.close()
            self._h5 = h5py.File(self.h5_path, "r")
            self._h5_pid = pid
        return self._h5

    def _validate(self) -> None:
        required = {"Image Index", "Finding Labels", "Patient ID"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en CSV: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row        = self.df.iloc[idx]
        image_name = row["Image Index"]
        label_vec  = self._labels[idx]  # (14,) float32

        arr = self._get_h5()["images"][image_name][:]  # (256, 256) uint8

        if self.transform is not None:
            arr = self.transform(arr)

        return {
            "image":       arr,
            "label":       torch.tensor(label_vec, dtype=torch.float32),
            "image_index": image_name,
            "finding_labels": row["Finding Labels"],
        }

    def class_distribution(self) -> dict:
        """Cuenta imágenes positivas por clase."""
        counts = self._labels.sum(axis=0).astype(int)
        return {CLASSES_14[i]: int(counts[i]) for i in range(NUM_CLASSES)}

    def compute_pos_weight(self, device: torch.device) -> torch.Tensor:
        """
        Calcula pos_weight para BCEWithLogitsLoss.
        pos_weight[i] = (N - N_pos[i]) / N_pos[i]

        Cuanto más rara es una clase, mayor es su peso.
        Esto compensa el desbalance sin necesidad de submuestreo.
        """
        N       = len(self._labels)
        n_pos   = self._labels.sum(axis=0).astype(float)
        n_pos   = np.maximum(n_pos, 1.0)  # evitar división por cero
        weights = (N - n_pos) / n_pos
        return torch.tensor(weights, dtype=torch.float32).to(device)

    def no_finding_count(self) -> int:
        """Imágenes sin ninguna patología."""
        return int((self._labels.sum(axis=1) == 0).sum())

    def __del__(self):
        if getattr(self, "_h5", None) is not None and self._h5.id.valid:
            self._h5.close()


def build_nih_multilabel_datasets(processed_dir: str,
                                   h5_path: str,
                                   train_transform=None,
                                   eval_transform=None) -> tuple:
    """Crea train/val/test datasets multi-label."""
    p = Path(processed_dir)
    h = Path(h5_path)
    return (
        NIHDatasetMultiLabel(p / "train.csv", h, train_transform),
        NIHDatasetMultiLabel(p / "val.csv",   h, eval_transform),
        NIHDatasetMultiLabel(p / "test.csv",  h, eval_transform),
    )


# ─── Smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Smoke test findings_to_vector:")

    cases = [
        ("No Finding",              [0]*14),
        ("Cardiomegaly",            None),
        ("Cardiomegaly|Effusion",   None),
        ("Infiltration|Pneumonia",  None),
    ]

    for finding, _ in cases:
        vec = findings_to_vector(finding)
        active = [CLASSES_14[i] for i in range(14) if vec[i] == 1]
        print(f"  '{finding}'")
        print(f"    → {vec.astype(int).tolist()}")
        print(f"    → Activas: {active if active else ['ninguna']}")
