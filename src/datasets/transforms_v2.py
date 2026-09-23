
import numpy as np
import torch
from torchvision import transforms


def build_train_transform_v2(image_size: int = 224):
    """Transform con augmentación. Evita XRayCenterCrop (no soporta 2D en versiones nuevas)."""
    return transforms.Compose([
        # (H,W) uint8 → (1,H,W) float tensor normalizado a [-1024, 1024]
        transforms.Lambda(lambda x: torch.from_numpy(
            x.astype(np.float32)).unsqueeze(0) / 255.0 * 2048.0 - 1024.0),
        transforms.CenterCrop(min(image_size * 256 // 224, 256)),
        transforms.Resize(image_size, antialias=True),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomAffine(degrees=10, translate=(0.07, 0.07), scale=(0.92, 1.08)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
    ])


def build_eval_transform_v2(image_size: int = 224):
    """Transform de evaluación sin augmentación."""
    return transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(
            x.astype(np.float32)).unsqueeze(0) / 255.0 * 2048.0 - 1024.0),
        transforms.CenterCrop(min(image_size * 256 // 224, 256)),
        transforms.Resize(image_size, antialias=True),
    ])
