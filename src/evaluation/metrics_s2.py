"""
src/evaluation/metrics_s2.py

Métricas de evaluación para clasificación multi-clase (Sprint 2).

Métricas principales:
  - AUC macro: promedio de AUC one-vs-rest por clase.
  - AUC por clase: AUC individual para cada clase.
  - Accuracy (top-1).
  - Precision, Recall (Sensitivity), Specificity — macro averaged.

La estrategia one-vs-rest (OvR) para AUC es el estándar en papers
de clasificación multi-clase de CXR (Wang 2017, Rajpurkar 2017).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize


CLASS_NAMES = {
    0: "No Finding",
    1: "Cardiomegaly",
    2: "Effusion",
    3: "Infiltration",
}


# ---------------------------------------------------------------------------
# Conversión de logits a probabilidades
# ---------------------------------------------------------------------------
def logits_to_probs_multiclass(logits: torch.Tensor) -> torch.Tensor:
    """
    Convierte logits (B, num_classes) a probabilidades via softmax.
    Devuelve tensor (B, num_classes).
    """
    return torch.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Métricas multi-clase
# ---------------------------------------------------------------------------
def compute_metrics_multiclass(y_true: np.ndarray | torch.Tensor,
                               y_prob: np.ndarray | torch.Tensor
                               ) -> dict:
    """
    Calcula métricas para clasificación multi-clase.

    Args:
        y_true: labels verdaderos (N,), enteros 0..C-1.
        y_prob: probabilidades softmax (N, C).

    Returns:
        dict con:
          auc_macro, auc_per_class,
          accuracy,
          precision_macro, sensitivity_macro, specificity_macro,
          confusion_matrix (C x C como lista).
    """
    y_true = _to_numpy(y_true).astype(int)
    y_prob = _to_numpy(y_prob).astype(float)

    n_classes = y_prob.shape[1]

    # AUC one-vs-rest por clase
    # Requiere que haya muestras de todas las clases presentes
    auc_per_class = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            # Solo una clase presente → AUC no definida
            auc_per_class.append(float('nan'))
        else:
            auc_per_class.append(float(roc_auc_score(y_bin, y_prob[:, c])))

    # AUC macro: promedio excluyendo NaN
    valid_aucs = [a for a in auc_per_class if not np.isnan(a)]
    auc_macro = float(np.mean(valid_aucs)) if valid_aucs else float('nan')

    # Predicción top-1
    y_pred = np.argmax(y_prob, axis=1)

    # Accuracy
    accuracy = float(accuracy_score(y_true, y_pred))

    # Métricas por clase (OvR) → promediar (macro)
    precisions, sensitivities, specificities = [], [], []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        tn = int(((y_pred != c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        precisions.append(prec)
        sensitivities.append(sens)
        specificities.append(spec)

    # Matriz de confusión
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return {
        'auc_macro': auc_macro,
        'auc_per_class': auc_per_class,
        'accuracy': accuracy,
        'precision_macro': float(np.mean(precisions)),
        'sensitivity_macro': float(np.mean(sensitivities)),
        'specificity_macro': float(np.mean(specificities)),
        'precision_per_class': precisions,
        'sensitivity_per_class': sensitivities,
        'specificity_per_class': specificities,
        'confusion_matrix': cm.tolist(),
        'n_samples': int(len(y_true)),
    }


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Formateo para logging
# ---------------------------------------------------------------------------
def format_metrics(metrics: dict, class_names: dict = CLASS_NAMES) -> str:
    lines = [
        f"AUC_macro={metrics['auc_macro']:.4f} | "
        f"Acc={metrics['accuracy']:.4f} | "
        f"Prec_macro={metrics['precision_macro']:.4f} | "
        f"Sens_macro={metrics['sensitivity_macro']:.4f}",
    ]
    for i, auc in enumerate(metrics['auc_per_class']):
        name = class_names.get(i, f"clase_{i}")
        lines.append(f"  AUC {name}: {auc:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n, c = 200, 4
    y_true = np.random.randint(0, c, size=n)
    y_prob = np.random.dirichlet(np.ones(c), size=n)
    # Mejorar un poco para que no sea puro ruido
    for i in range(n):
        y_prob[i, y_true[i]] += 0.5
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    m = compute_metrics_multiclass(y_true, y_prob)
    print("Smoke test metrics_s2.py:")
    print(format_metrics(m))
    print(f"\nMatriz de confusión:\n{np.array(m['confusion_matrix'])}")
