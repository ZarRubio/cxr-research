"""
src/datasets/transforms.py

Pipeline de preprocesamiento de imagenes para radiografias de torax.

Se usan los transforms nativos de torchxrayvision porque:
  1. XRayCenterCrop ajusta el aspect ratio al centro de la imagen (las CXR
     no siempre vienen cuadradas, y los bordes suelen ser ruido o letras).
  2. XRayResizer escala a 224x224 manteniendo las propiedades del modelo
     preentrenado (los modelos 'res224' de xrv esperan ese tamano).

Ademas se agregan:
  - Normalizacion al rango [-1024, 1024] que es el formato esperado por
    los modelos de xrv (las CXR originales en 16-bit se normalizan asi).
  - Data augmentation opcional (solo en entrenamiento): flips, rotaciones
    pequenas, jitter de brillo. Se mantiene conservador porque en CXR un
    flip horizontal puede alterar la anatomia (el corazon esta a la izquierda
    del paciente, no del observador).

IMPORTANTE sobre flips:
  - El flip horizontal INVIERTE la posicion del corazon. Para clasificacion
    de cardiomegalia esto podria enganar al modelo. Por defecto se DESACTIVA.
  - Se usan solo rotaciones chicas (+-5 grados) y jitter leve de brillo.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
import torchxrayvision as xrv


# ---------------------------------------------------------------------------
# Normalizacion
# ---------------------------------------------------------------------------
class NormalizeForXRV:
    """
    Normaliza imagenes 8-bit (0-255) al rango esperado por los modelos xrv:
    [-1024, 1024], float32.

    Formula: xrv.datasets.normalize(img_uint8, maxval=255).

    Las imagenes NIH vienen como PNG 8-bit, pero los modelos xrv fueron
    entrenados con CXR 16-bit. Esta funcion hace el puente.
    """
    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: np.ndarray uint8 (H, W) o (1, H, W), valores 0-255
        return xrv.datasets.normalize(img, maxval=255, reshape=True)


# ---------------------------------------------------------------------------
# Augmentations (solo para train)
# ---------------------------------------------------------------------------
class RandomRotationSmall:
    """Rotacion aleatoria de +-max_degrees grados."""
    def __init__(self, max_degrees: float = 5.0):
        self.max_degrees = max_degrees

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # img: (1, H, W) float32
        from scipy.ndimage import rotate
        angle = np.random.uniform(-self.max_degrees, self.max_degrees)
        rotated = rotate(img, angle, axes=(1, 2), reshape=False,
                         order=1, mode='nearest')
        return rotated.astype(np.float32)


class RandomBrightnessJitter:
    """Escala la intensidad en [1-factor, 1+factor]."""
    def __init__(self, factor: float = 0.1):
        self.factor = factor

    def __call__(self, img: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(1.0 - self.factor, 1.0 + self.factor)
        return (img * scale).astype(np.float32)


class ToTensor:
    """np.ndarray (1, H, W) -> torch.Tensor."""
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).float()


# ---------------------------------------------------------------------------
# Pipelines finales
# ---------------------------------------------------------------------------
def build_transform(image_size: int = 224, augment: bool = False
                    ) -> T.Compose:
    """
    Construye el pipeline de preprocesamiento.

    Args:
        image_size: resolucion objetivo (224 para modelos *-res224).
        augment: si True, agrega augmentations ligeras (solo para train).

    Returns:
        torchvision.transforms.Compose aplicable a una imagen numpy (1, H, W).
    """
    steps = [
        # 1. Normalizar a rango xrv (mantiene shape (1, H, W))
        NormalizeForXRV(),
        # 2. Center crop + resize (xrv spera input (1, H, W))
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(image_size),
    ]

    if augment:
        steps.extend([
            RandomRotationSmall(max_degrees=5.0),
            RandomBrightnessJitter(factor=0.1),
        ])

    # 3. Convertir a tensor PyTorch
    steps.append(ToTensor())

    return T.Compose(steps)


def build_train_transform(image_size: int = 224) -> T.Compose:
    """Pipeline para el split de train (con augmentation)."""
    return build_transform(image_size=image_size, augment=True)


def build_eval_transform(image_size: int = 224) -> T.Compose:
    """Pipeline para val/test (sin augmentation)."""
    return build_transform(image_size=image_size, augment=False)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Imagen sintetica 8-bit (1024x1024, como vienen las NIH)
    fake_img = np.random.randint(0, 256, size=(1024, 1024), dtype=np.uint8)

    print("Test eval transform:")
    tfm_eval = build_eval_transform(image_size=224)
    out = tfm_eval(fake_img)
    print(f"  shape: {out.shape}, dtype: {out.dtype}")
    print(f"  rango: [{out.min():.2f}, {out.max():.2f}]")

    print("\nTest train transform (con augmentation):")
    tfm_train = build_train_transform(image_size=224)
    out = tfm_train(fake_img)
    print(f"  shape: {out.shape}, dtype: {out.dtype}")
    print(f"  rango: [{out.min():.2f}, {out.max():.2f}]")

    print("\nOK: transforms.py funciona correctamente.")
