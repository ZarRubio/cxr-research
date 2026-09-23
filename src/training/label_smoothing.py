"""
src/training/label_smoothing.py
Label Smoothing Cross-Entropy para Sprint 4 (Mejora B).

En vez de targets duros [0,0,1,0], usa targets suaves:
  [ε/(C-1), ε/(C-1), 1-ε, ε/(C-1)]  con ε=0.1, C=4

Beneficios:
  1. Reduce overconfidence del modelo.
  2. Mejora calibración de probabilidades.
  3. Actúa como regularizador adicional.

Referencia: Szegedy et al. (2016). Rethinking the Inception Architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 weight: torch.Tensor | None = None):
        super().__init__()
        assert 0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)
        log_prob  = F.log_softmax(pred, dim=1)

        # Targets suaves
        smooth_val = self.smoothing / (n_classes - 1)
        with torch.no_grad():
            smooth_target = torch.full_like(log_prob, smooth_val)
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # Loss por muestra
        loss = -(smooth_target * log_prob).sum(dim=1)  # (B,)

        # Aplicar class weights si se proporcionaron
        if self.weight is not None:
            w = self.weight.to(pred.device)[target]
            loss = (loss * w).mean()
        else:
            loss = loss.mean()

        return loss
