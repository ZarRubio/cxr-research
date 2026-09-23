"""
src/evaluation/metrics.py

Metricas de evaluacion para clasificacion binaria de CXR.

Metricas implementadas (alineadas con el Project Charter):
  - AUC (area bajo la curva ROC)
  - Accuracy
  - Precision (VP / (VP + FP))
  - Sensitivity / Recall / True Positive Rate (VP / (VP + FN))
  - Specificity / True Negative Rate (VN / (VN + FP))

Convenciones:
  - Clase positiva: label 1 (Cardiomegaly)
  - Clase negativa: label 0 (No Finding)
  - AUC se calcula sobre probabilidades (no sobre predicciones binarias).
  - Las demas metricas requieren umbralizar las probabilidades (default 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Estructura de resultados
# ---------------------------------------------------------------------------
@dataclass
class ClassificationMetrics:
    """Contenedor tipado para las 5 metricas + conteos de confusion matrix."""
    auc: float
    accuracy: float
    precision: float
    sensitivity: float  # = recall = TPR
    specificity: float  # = TNR
    # Conteos de confusion matrix (utiles para debug)
    tp: int
    fp: int
    tn: int
    fn: int
    # Metadata
    threshold: float
    n_samples: int

    def as_dict(self) -> dict:
        return asdict(self)

    def as_short_dict(self) -> dict:
        """Solo las 5 metricas principales, en porcentaje."""
        return {
            'auc': self.auc,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
        }

    def __str__(self) -> str:
        return (
            f"AUC={self.auc:.4f} | "
            f"Acc={self.accuracy:.4f} | "
            f"Prec={self.precision:.4f} | "
            f"Sens={self.sensitivity:.4f} | "
            f"Spec={self.specificity:.4f} "
            f"(TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn})"
        )


# ---------------------------------------------------------------------------
# Calculo de metricas
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray | torch.Tensor,
                    y_prob: np.ndarray | torch.Tensor,
                    threshold: float = 0.5) -> ClassificationMetrics:
    """
    Calcula las 5 metricas para clasificacion binaria.

    Args:
        y_true: labels verdaderos (0 o 1), shape (N,).
        y_prob: probabilidad de la clase positiva, shape (N,) en [0, 1].
        threshold: umbral para binarizar probabilidades (default 0.5).

    Returns:
        ClassificationMetrics con las 5 metricas y conteos.
    """
    y_true = _to_numpy(y_true).astype(int)
    y_prob = _to_numpy(y_prob).astype(float)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shapes distintos: y_true={y_true.shape}, "
                         f"y_prob={y_prob.shape}")

    # Binarizar con el umbral
    y_pred = (y_prob >= threshold).astype(int)

    # AUC requiere al menos ambas clases presentes
    if len(np.unique(y_true)) < 2:
        # Edge case: todos los labels son iguales (raro pero posible en batches).
        auc = float('nan')
    else:
        auc = roc_auc_score(y_true, y_prob)

    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix. Fijamos labels=[0, 1] para garantizar shape 2x2
    # incluso si el batch no tiene ambas clases.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Precision: VP / (VP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Sensitivity / Recall: VP / (VP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity: VN / (VN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return ClassificationMetrics(
        auc=float(auc),
        accuracy=float(accuracy),
        precision=float(precision),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        threshold=float(threshold),
        n_samples=int(len(y_true)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    """Convierte torch.Tensor o np.ndarray a np.ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def logits_to_probs(logits: torch.Tensor, positive_class: int = 1
                    ) -> torch.Tensor:
    """
    Convierte logits (B, num_classes) a probabilidad de la clase positiva (B,).

    Default positive_class=1 (Cardiomegaly en nuestra convencion).
    """
    probs = torch.softmax(logits, dim=1)
    return probs[:, positive_class]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Caso sintetico: 100 muestras, 30% positivas, modelo razonablemente bueno
    np.random.seed(42)
    n = 100
    y_true = np.random.binomial(1, 0.3, size=n)
    # Probabilidades correlacionadas con y_true (modelo con 85% de acierto aprox)
    y_prob = np.where(y_true == 1,
                      np.random.beta(5, 2, size=n),      # positivos -> probs altas
                      np.random.beta(2, 5, size=n))      # negativos -> probs bajas

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    print("Smoke test metrics.py:")
    print(metrics)
    print()
    print("As dict:", metrics.as_short_dict())
