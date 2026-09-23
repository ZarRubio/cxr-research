"""
src/models/baseline.py  (v1.1 - fix de relu)

Modelo baseline del Sprint 1: DenseNet121 preentrenado en NIH ChestX-ray14
(via torchxrayvision) adaptado a clasificacion binaria.

FIX v1.1: cambio `torch.relu(x, inplace=False)` por `torch.relu(x)`.
`torch.relu` no acepta `inplace` (solo `torch.nn.functional.relu` lo hace).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv


class BaselineCNN(nn.Module):
    """
    DenseNet121 de torchxrayvision con cabeza de clasificacion binaria.

    Args:
        backbone_weights: string del modelo preentrenado en xrv
                          (default: "densenet121-res224-nih").
        num_classes: numero de clases de salida (2 para binaria).
        finetune_mode: "head" (solo capa final) o "full" (todo el modelo).
    """

    def __init__(self, backbone_weights: str = "densenet121-res224-nih",
                 num_classes: int = 2, finetune_mode: str = "head"):
        super().__init__()

        if finetune_mode not in {"head", "full"}:
            raise ValueError(f"finetune_mode debe ser 'head' o 'full', "
                             f"no '{finetune_mode}'")
        self.finetune_mode = finetune_mode
        self.num_classes = num_classes

        # Cargar DenseNet121 preentrenado de xrv
        self.backbone = xrv.models.DenseNet(weights=backbone_weights)

        # Dim de features (1024 para DenseNet121)
        num_features = self.backbone.classifier.in_features

        # Reemplazar classifier (14 clases multi-label) por uno binario
        self.backbone.classifier = nn.Linear(num_features, num_classes)

        self._apply_finetune_mode()

    def _apply_finetune_mode(self) -> None:
        if self.finetune_mode == "head":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        else:  # full
            for param in self.backbone.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Evitamos la sigmoid interna de xrv llamando los pasos
        manualmente y devolvemos logits crudos.
        """
        features = self.backbone.features(x)
        out = F.relu(features, inplace=False)  # <-- FIX: usar F.relu, no torch.relu
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = self.backbone.classifier(out)
        return logits

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_baseline_model(cfg) -> BaselineCNN:
    return BaselineCNN(
        backbone_weights=cfg.model['backbone'],
        num_classes=cfg.model['num_classes'],
        finetune_mode=cfg.model['finetune_mode'],
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path

    code_dir = Path(__file__).resolve().parent.parent.parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from src.training.utils import load_config, get_device

    cfg = load_config(code_dir / "configs" / "baseline.yaml")
    device = get_device()

    model = build_baseline_model(cfg).to(device)
    print(f"backbone: {cfg.model['backbone']}")
    print(f"num_classes: {cfg.model['num_classes']}")
    print(f"finetune_mode: {cfg.model['finetune_mode']}")
    print(f"parametros totales:     {model.total_params():>12,}")
    print(f"parametros entrenables: {model.trainable_params():>12,}")

    x = torch.randn(4, 1, 224, 224, device=device) * 500
    logits = model(x)
    print(f"\nForward OK: input={x.shape} -> output={logits.shape}")
    probs = torch.softmax(logits, dim=1)
    print(f"probs[0]: {probs[0].tolist()}")
