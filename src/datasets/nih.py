"""
src/datasets/nih.py  (v1.1 - fix de normalizacion)

Dataset de PyTorch para el baseline binario NIH (No Finding vs Cardiomegaly).

FIX en v1.1:
  Se elimino el np.newaxis al cargar la imagen porque entraba en conflicto
  con reshape=True de xrv.datasets.normalize (doble reshape -> shape (1,1,H,W)
  que colapsaba la imagen a un valor constante).

  Ahora la imagen se carga como (H, W) uint8 y el primer transform
  (NormalizeForXRV) se encarga de agregar la dimension de canal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class NIHDataset(Dataset):
    """
    Dataset NIH para clasificacion binaria.

    Args:
        csv_path: ruta al CSV generado por prepare_splits (train/val/test).
        images_dir: carpeta que contiene los archivos .png del NIH.
        transform: callable a aplicar sobre cada imagen (de transforms.py).

    Cada __getitem__ retorna:
        {
          'image': torch.Tensor (1, H, W), float32 en rango [-1024, 1024],
          'label': torch.Tensor escalar, long (0 o 1),
          'image_index': str (nombre del archivo, util para debugging),
        }
    """

    def __init__(self, csv_path: str | Path, images_dir: str | Path,
                 transform=None):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV no encontrado: {self.csv_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images_dir no encontrado: {self.images_dir}")

        self.df = pd.read_csv(self.csv_path)
        self._validate_columns()

    def _validate_columns(self) -> None:
        required = {'Image Index', 'label', 'Patient ID'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Columnas faltantes en {self.csv_path}: {missing}")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_name = row['Image Index']
        label = int(row['label'])

        image = self._load_image(image_name)

        if self.transform is not None:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_index': image_name,
        }

    # ------------------------------------------------------------------
    def _load_image(self, image_name: str) -> np.ndarray:
        """
        Carga una imagen como array 2D (H, W) uint8 en escala de grises.

        FIX v1.1: NO agregamos np.newaxis aqui. El primer transform
        (NormalizeForXRV, con reshape=True) se encarga de convertir
        (H, W) -> (1, H, W). Si agregaramos newaxis aqui, el reshape
        lo duplicaria y romperia el pipeline.
        """
        img_path = self.images_dir / image_name
        if not img_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert('L')  # forzar escala de grises
            arr = np.array(pil_img, dtype=np.uint8)

        # Devolver como (H, W). El transform agrega la dim de canal.
        return arr

    # ------------------------------------------------------------------
    def class_distribution(self) -> dict:
        counts = self.df['label'].value_counts().to_dict()
        return {
            'positive': int(counts.get(1, 0)),
            'negative': int(counts.get(0, 0)),
            'total': len(self.df),
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
def build_nih_datasets(cfg, train_transform, eval_transform
                       ) -> tuple[NIHDataset, NIHDataset, NIHDataset]:
    """
    Crea los tres datasets (train, val, test) usando la config del proyecto.
    """
    processed = Path(cfg.paths['data_processed'])
    images_dir = Path(cfg.paths['data_raw']) / cfg.dataset['images_subdir']

    train_ds = NIHDataset(
        csv_path=processed / "train.csv",
        images_dir=images_dir,
        transform=train_transform,
    )
    val_ds = NIHDataset(
        csv_path=processed / "val.csv",
        images_dir=images_dir,
        transform=eval_transform,
    )
    test_ds = NIHDataset(
        csv_path=processed / "test.csv",
        images_dir=images_dir,
        transform=eval_transform,
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    code_dir = Path(__file__).resolve().parent.parent.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from src.training.utils import load_config
    from src.datasets.transforms import build_train_transform, build_eval_transform

    cfg = load_config(code_dir / "configs" / "baseline.yaml")

    train_tfm = build_train_transform(image_size=cfg.preprocessing['image_size'])
    eval_tfm = build_eval_transform(image_size=cfg.preprocessing['image_size'])

    train_ds, val_ds, test_ds = build_nih_datasets(cfg, train_tfm, eval_tfm)

    print(f"Train: {len(train_ds)} muestras | dist: {train_ds.class_distribution()}")
    print(f"Val:   {len(val_ds)} muestras | dist: {val_ds.class_distribution()}")
    print(f"Test:  {len(test_ds)} muestras | dist: {test_ds.class_distribution()}")

    sample = train_ds[0]
    print(f"\nMuestra [0]:")
    print(f"  image shape: {sample['image'].shape}")
    print(f"  image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    print(f"  label: {sample['label'].item()}")

def build_nih_datasets_from_processed(
        processed_dir: str,
        images_dir: str,
        train_transform=None,
        eval_transform=None,
) -> tuple:
    """
    Crea los tres datasets leyendo CSVs desde un directorio de splits
    personalizado (permite Sprint 1 y Sprint 2 coexistir).
    """
    from pathlib import Path as _Path
    processed = _Path(processed_dir)
    imgs = _Path(images_dir)
    train_ds = NIHDataset(processed / "train.csv", imgs, train_transform)
    val_ds   = NIHDataset(processed / "val.csv",   imgs, eval_transform)
    test_ds  = NIHDataset(processed / "test.csv",  imgs, eval_transform)
    return train_ds, val_ds, test_ds
