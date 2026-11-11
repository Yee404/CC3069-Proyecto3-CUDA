"""Métricas de clasificación del notebook 04, compartidas por CPU y GPU.

Copia exacta de `metricas()` y `umbral_optimo()` de `04_modelos_cianobacteria.ipynb`
(rama `parte-2`, commit `6014d7a`), para que las versiones CPU y GPU se evalúen con
el mismo código que el laboratorio. Solo usa NumPy y scikit-learn (CPU): las
probabilidades que vienen de la GPU se copian al host antes de evaluar.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metricas(nombre, y_real, probabilidad, umbral=0.5, extra=None) -> dict:
    """Accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC y matriz de confusión."""
    prediccion = (probabilidad >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, prediccion, labels=[0, 1]).ravel()
    fila = {
        "modelo": nombre,
        "umbral": round(float(umbral), 4),
        "accuracy": accuracy_score(y_real, prediccion),
        "precision": precision_score(y_real, prediccion, zero_division=0),
        "recall": recall_score(y_real, prediccion, zero_division=0),
        "f1": f1_score(y_real, prediccion, zero_division=0),
        "f2": fbeta_score(y_real, prediccion, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_real, probabilidad) if len(set(y_real)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_real, probabilidad) if len(set(y_real)) > 1 else np.nan,
        "VN": tn, "FP": fp, "FN": fn, "VP": tp,
    }
    if extra:
        fila.update(extra)
    return fila


def umbral_optimo(y_real, probabilidad, beta=2.0) -> tuple[float, float]:
    """Umbral que maximiza el F-beta sobre la curva de precision-recall."""
    precision, recall, umbrales = precision_recall_curve(y_real, probabilidad)
    b2 = beta ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1 + b2) * precision * recall / (b2 * precision + recall)
    f = np.nan_to_num(f[:-1])
    indice = int(np.argmax(f))
    return float(umbrales[indice]), float(f[indice])
