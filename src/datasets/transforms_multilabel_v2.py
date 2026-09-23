"""
src/datasets/transforms_multilabel_v2.py
Augmentación para Sprint 4 multi-label v2.
Aplica augmentación DESPUÉS de convertir a tensor para evitar
problemas con XRayCenterCrop en imágenes 2D del HDF5.
"""
import numpy as np
import torch
from torchvision import transforms
import torchxrayvision as xrv


class ArrayToXRVTensor:
    """(H,W) uint8 → (1,H,H) float32 en [-1024,1024] (formato xrv)."""
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32)
        img = (img / 255.0) * 2048.0 - 1024.0
        return torch.from_numpy(img).unsqueeze(0)


class GaussianNoise:
    """Agrega ruido gaussiano para simular variaciones del equipo RX."""
    def __init__(self, sigma: float = 20.0):
        self.sigma = sigma

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.sigma > 0:
            noise = torch.randn_like(tensor) * self.sigma
            return (tensor + noise).clamp(-1024, 1024)
        return tensor


def build_train_transform_v2(image_size: int = 224):
    """Transform con augmentación — compatible con HDF5 arrays 2D."""
    return transforms.Compose([
        # 1. Resize a imagen cuadrada
        transforms.Lambda(
            lambda x: torch.from_numpy(
                __import__('cv2').resize(
                    x.astype(np.float32), (image_size, image_size)
                )
            ).unsqueeze(0) / 255.0 * 2048.0 - 1024.0
        ),
        # 2. Augmentaciones espaciales (sobre tensor)
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.07, 0.07),
            scale=(0.92, 1.08),
        ),
        # 3. Ruido gaussiano
        GaussianNoise(sigma=20.0),
    ])


def build_eval_transform_v2(image_size: int = 224):
    """Transform de evaluación sin augmentación."""
    return transforms.Compose([
        transforms.Lambda(
            lambda x: torch.from_numpy(
                __import__('cv2').resize(
                    x.astype(np.float32), (image_size, image_size)
                )
            ).unsqueeze(0) / 255.0 * 2048.0 - 1024.0
        ),
    ])
