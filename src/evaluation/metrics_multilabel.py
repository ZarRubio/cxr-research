"""
src/evaluation/metrics_multilabel.py

Métricas para clasificación multi-label (Sprint 4 - 14 clases).

Métricas principales:
  - AUC por clase (one-vs-rest, estándar en NIH CXR papers)
  - AUC macro (promedio de las 14 clases)
  - mAP (mean Average Precision)
  - Accuracy por clase (con threshold 0.5)

Referencia: Wang et al. 2017 (ChestX-ray14) usa AUC por clase.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

CLASSES_14 = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion",    "Emphysema",    "Fibrosis",      "Hernia",
    "Infiltration","Mass",         "Nodule",         "Pleural_Thickening",
    "Pneumonia",   "Pneumothorax",
]


def compute_metrics_multilabel(y_true:  np.ndarray,
                                y_probs: np.ndarray,
                                threshold: float = 0.5) -> dict:
    """
    Calcula métricas multi-label.

    Args:
        y_true:  (N, 14) float32 — labels binarios ground truth.
        y_probs: (N, 14) float32 — probabilidades sigmoid del modelo.
        threshold: para calcular accuracy/precision/recall por clase.

    Returns:
        dict con auc_per_class, auc_macro, map, accuracy_per_class, etc.
    """
    y_true  = _to_numpy(y_true)
    y_probs = _to_numpy(y_probs)
    y_pred  = (y_probs >= threshold).astype(int)

    n_classes   = y_true.shape[1]
    auc_list    = []
    ap_list     = []
    acc_list    = []
    sens_list   = []
    spec_list   = []

    for c in range(n_classes):
        y_c    = y_true[:, c]
        p_c    = y_probs[:, c]
        pred_c = y_pred[:, c]

        n_pos = y_c.sum()
        n_neg = len(y_c) - n_pos

        # AUC — solo si hay positivos y negativos
        if n_pos > 0 and n_neg > 0:
            auc = float(roc_auc_score(y_c, p_c))
            ap  = float(average_precision_score(y_c, p_c))
        else:
            auc = float("nan")
            ap  = float("nan")

        auc_list.append(auc)
        ap_list.append(ap)

        # Accuracy, Sensitivity, Specificity con threshold
        tp = int(((pred_c == 1) & (y_c == 1)).sum())
        fp = int(((pred_c == 1) & (y_c == 0)).sum())
        tn = int(((pred_c == 0) & (y_c == 0)).sum())
        fn = int(((pred_c == 0) & (y_c == 1)).sum())

        acc  = (tp + tn) / (tp + fp + tn + fn) if (tp+fp+tn+fn) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        acc_list.append(float(acc))
        sens_list.append(float(sens))
        spec_list.append(float(spec))

    # Promedios excluyendo NaN
    valid_auc = [a for a in auc_list if not np.isnan(a)]
    valid_ap  = [a for a in ap_list  if not np.isnan(a)]

    return {
        "auc_per_class":      auc_list,          # lista de 14
        "auc_macro":          float(np.mean(valid_auc)) if valid_auc else float("nan"),
        "ap_per_class":       ap_list,            # lista de 14
        "map":                float(np.mean(valid_ap))  if valid_ap  else float("nan"),
        "accuracy_per_class": acc_list,
        "accuracy_macro":     float(np.mean(acc_list)),
        "sensitivity_per_class": sens_list,
        "sensitivity_macro":  float(np.mean(sens_list)),
        "specificity_per_class": spec_list,
        "specificity_macro":  float(np.mean(spec_list)),
        "n_samples":          int(len(y_true)),
        "n_classes":          n_classes,
        "threshold":          threshold,
    }


def format_metrics_multilabel(metrics: dict) -> str:
    """Formatea las métricas para logging."""
    lines = [
        f"AUC macro={metrics['auc_macro']:.4f} | "
        f"mAP={metrics['map']:.4f} | "
        f"Acc macro={metrics['accuracy_macro']:.4f}",
        "",
        f"{'Clase':<22} {'AUC':>6} {'AP':>6} {'Sens':>6} {'Spec':>6}",
        "-" * 50,
    ]
    for i, cls in enumerate(CLASSES_14):
        auc  = metrics["auc_per_class"][i]
        ap   = metrics["ap_per_class"][i]
        sens = metrics["sensitivity_per_class"][i]
        spec = metrics["specificity_per_class"][i]
        auc_str  = f"{auc:.4f}"  if not np.isnan(auc) else "  N/A"
        ap_str   = f"{ap:.4f}"   if not np.isnan(ap)  else "  N/A"
        lines.append(
            f"{cls:<22} {auc_str:>6} {ap_str:>6} "
            f"{sens:>6.4f} {spec:>6.4f}"
        )
    return "\n".join(lines)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ─── Smoke test ───────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N, C = 200, 14
    y_true  = (np.random.rand(N, C) > 0.85).astype(np.float32)
    y_probs = np.clip(y_true + np.random.randn(N, C) * 0.3, 0, 1)

    m = compute_metrics_multilabel(y_true, y_probs)
    print("Smoke test metrics_multilabel.py:")
    print(format_metrics_multilabel(m))
    print(f"\nAUC macro: {m['auc_macro']:.4f}")
    print(f"mAP:       {m['map']:.4f}")
