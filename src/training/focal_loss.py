"""
src/training/focal_loss.py
Focal Loss para clasificación multi-clase con clases desbalanceadas.

FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

  - gamma > 0: reduce la pérdida de ejemplos bien clasificados (fáciles),
    forzando al modelo a enfocarse en los difíciles (clases raras).
  - alpha: pesos por clase (igual que class_weights en CrossEntropy).

Valores típicos en imágenes médicas:
  gamma = 2.0 (valor original del paper RetinaNet, Lin et al. 2017)
  alpha = inversamente proporcional a la frecuencia de clase
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0,
                 alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha      # tensor (num_classes,) o None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) — logits crudos del modelo
            targets: (B,)   — labels enteros 0..C-1
        """
        # Probabilidades via softmax
        log_prob = F.log_softmax(logits, dim=1)          # (B, C)
        prob     = log_prob.exp()                         # (B, C)

        # Probabilidad de la clase verdadera
        # gather: selecciona prob[i, targets[i]] para cada i
        pt = prob.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Factor focal: (1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma          # (B,)

        # Log-prob de la clase verdadera
        log_pt = log_prob.gather(
            dim=1, index=targets.unsqueeze(1)
        ).squeeze(1)                                      # (B,)

        # Loss base: -log(pt)
        loss = -focal_weight * log_pt                     # (B,)

        # Aplicar alpha si se proporcionó
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
