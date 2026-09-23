"""
src/explainability/gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) para el modelo baseline.

Grad-CAM ilumina las regiones de la imagen que mas contribuyen a la prediccion
de una clase especifica. Para cardiomegalia, esperamos que el mapa de calor
ilumine la region cardiaca (silueta del corazon).

Implementacion: usamos la libreria `pytorch-grad-cam` que ya esta en requirements.

Target layer: la ultima conv del backbone DenseNet121.
  En xrv.models.DenseNet la estructura es:
    backbone.features.denseblock4.denselayer16.conv2  (ultima conv)
  Pero lo mas comun es apuntar al ultimo denseblock entero:
    backbone.features.denseblock4
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model: nn.Module, target_layer_name: str = "features.denseblock4"):
    """
    Navega la jerarquia del modelo para encontrar la capa objetivo.

    Ejemplo: "features.denseblock4" -> model.backbone.features.denseblock4
    """
    # El modelo tiene backbone adentro
    module = model.backbone
    for part in target_layer_name.split('.'):
        module = getattr(module, part)
    return module


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen con rango [-1024, 1024] (formato xrv) a [0, 1]
    para poder superponerla con el mapa de calor.
    """
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def compute_gradcam_for_sample(model: nn.Module, image_tensor: torch.Tensor,
                                target_class: int,
                                target_layer) -> np.ndarray:
    """
    Calcula Grad-CAM para una imagen.

    Args:
        model: modelo entrenado.
        image_tensor: tensor (1, H, W) o (1, 1, H, W) en formato xrv.
        target_class: clase objetivo (0 o 1).
        target_layer: capa del modelo a visualizar.

    Returns:
        heatmap: np.ndarray (H, W) en [0, 1].
    """
    # Asegurar shape (1, 1, H, W) para batch
    if image_tensor.ndim == 3:
        input_tensor = image_tensor.unsqueeze(0)
    else:
        input_tensor = image_tensor

    input_tensor = input_tensor.requires_grad_(True)

    # pytorch-grad-cam espera que el modelo devuelva logits
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0]  # (H, W)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Superpone el heatmap sobre la imagen en escala de grises.

    Args:
        image: (H, W) en [0, 1] (imagen base).
        heatmap: (H, W) en [0, 1] (Grad-CAM).
        alpha: peso del heatmap (0=solo imagen, 1=solo heatmap).

    Returns:
        RGB image (H, W, 3) en [0, 1].
    """
    # Imagen base en escala de grises repetida a 3 canales
    img_rgb = np.stack([image] * 3, axis=-1)

    # Heatmap en jet colormap
    cmap = plt.get_cmap('jet')
    heatmap_rgb = cmap(heatmap)[..., :3]  # descartar alpha

    overlay = (1 - alpha) * img_rgb + alpha * heatmap_rgb
    return np.clip(overlay, 0, 1)


def visualize_gradcam_grid(model: nn.Module, dataset, device: torch.device,
                            n_positive: int = 3, n_negative: int = 3,
                            target_layer_name: str = "features.denseblock4",
                            output_path: Path | str | None = None) -> None:
    """
    Selecciona algunas imagenes del dataset, calcula Grad-CAM y las visualiza
    en un grid 2xN (fila 1: positivos, fila 2: negativos).
    """
    model.eval()
    target_layer = get_target_layer(model, target_layer_name)

    # Buscar indices de positivos y negativos en el dataset
    labels_series = dataset.df['label']
    pos_indices = labels_series[labels_series == 1].index.tolist()[:n_positive]
    neg_indices = labels_series[labels_series == 0].index.tolist()[:n_negative]

    n_cols = max(n_positive, n_negative)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    for row, (title, indices) in enumerate([
        ("Cardiomegaly (label=1)", pos_indices),
        ("No Finding (label=0)", neg_indices),
    ]):
        for col in range(n_cols):
            ax = axes[row, col] if n_cols > 1 else axes[row]
            if col >= len(indices):
                ax.axis('off')
                continue

            idx = indices[col]
            sample = dataset[idx]
            image_tensor = sample['image'].to(device)
            true_label = sample['label'].item()
            image_name = sample['image_index']

            # Prediccion del modelo
            with torch.no_grad():
                logits = model(image_tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                pred_prob = probs[0, 1].item()
                pred_label = int(pred_prob >= 0.5)

            # Grad-CAM para la clase verdadera
            heatmap = compute_gradcam_for_sample(
                model, image_tensor, target_class=true_label,
                target_layer=target_layer,
            )

            # Superponer
            img_np = image_tensor.squeeze().cpu().numpy()
            img_norm = normalize_for_display(img_np)
            overlay = overlay_heatmap(img_norm, heatmap, alpha=0.45)

            ax.imshow(overlay)
            correct = (pred_label == true_label)
            marker = "✓" if correct else "✗"
            color = 'green' if correct else 'red'
            ax.set_title(
                f"{image_name}\ntrue={true_label} | pred={pred_label} "
                f"(p={pred_prob:.2f}) {marker}",
                fontsize=9, color=color,
            )
            ax.axis('off')

        # Label de fila
        if n_cols > 0:
            axes[row, 0].set_ylabel(title, fontsize=12, rotation=90,
                                     labelpad=20, fontweight='bold')

    plt.suptitle("Grad-CAM sobre test set (colores calidos = regiones importantes)",
                 fontsize=13, y=1.00)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        print(f"Grad-CAM guardado en: {output_path}")

    plt.show()
