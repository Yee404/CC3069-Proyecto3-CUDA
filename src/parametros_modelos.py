"""Hiperparámetros compartidos de la Parte II (CPU y GPU usan exactamente estos).

Regresión Logística: la rejilla del notebook 04.
XGBoost: reemplazo de `HistGradientBoostingClassifier` (HGB) con la equivalencia de
`CONTEXTO_INFORME.md` §2.2. Entre CPU y GPU solo cambia `device` (y `n_jobs`).

    from src.parametros_modelos import REJILLA_XGBOOST, parametros_xgboost, peso_positivos

    params = parametros_xgboost(learning_rate=0.1, max_leaves=31,
                                scale_pos_weight=peso_positivos(y_ent), device="cuda")
    modelo = xgboost.XGBClassifier(**params).fit(X_ent, y_ent)
"""

from __future__ import annotations

from itertools import product

import numpy as np

SEMILLA = 42

# --------------------------------------------------------------------------
# Regresión Logística (notebook 04, Ej. 4)
# --------------------------------------------------------------------------
REJILLA_LOGISTICA = {"C": [0.1, 1.0, 10.0]}
PARAMETROS_LOGISTICA = {"class_weight": "balanced", "max_iter": 1000, "random_state": SEMILLA}

# --------------------------------------------------------------------------
# XGBoost (equivalente a HistGradientBoostingClassifier)
# --------------------------------------------------------------------------
# Rejilla equivalente a {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}.
REJILLA_XGBOOST = {"learning_rate": [0.05, 0.1], "max_leaves": [31, 63]}

# Hiperparámetros fijos. Cada uno replica un valor por defecto de HGB:
PARAMETROS_XGBOOST_BASE = {
    "objective": "binary:logistic",
    "tree_method": "hist",        # histogramas, como HGB
    "grow_policy": "lossguide",   # crece hoja por hoja (best-first), como HGB
    "max_depth": 0,               # sin límite de profundidad (HGB: max_depth=None);
                                  # OJO: el defecto de XGBoost es 6 aun con lossguide
    "n_estimators": 100,          # HGB: max_iter=100
    "max_bin": 256,               # HGB: max_bins=255 (+1 bin reservado a faltantes)
    "reg_lambda": 0.0,            # HGB: l2_regularization=0 (XGBoost usa 1 por defecto)
    "min_child_weight": 1.0,      # HGB: min_samples_leaf=20 (no hay equivalente exacto)
    "eval_metric": "logloss",     # HGB: scoring="loss" para el early stopping
    "random_state": SEMILLA,
}

# Early stopping de HGB con early_stopping="auto" y más de 10 000 filas.
FRACCION_VALIDACION_ES = 0.1
RONDAS_SIN_MEJORA_ES = 10


def peso_positivos(y: np.ndarray) -> float:
    """scale_pos_weight = negativos / positivos (equivale a class_weight="balanced").

    Se calcula con las etiquetas del conjunto con el que se entrena cada ajuste,
    igual que HGB calcula `class_weight="balanced"` dentro de cada `fit`.
    """
    positivos = int(np.sum(y == 1))
    return float((len(y) - positivos) / positivos)


def parametros_xgboost(
    learning_rate: float,
    max_leaves: int,
    scale_pos_weight: float,
    device: str = "cpu",
    n_jobs: int | None = -1,
    early_stopping: bool = False,
) -> dict:
    """Diccionario para `xgboost.XGBClassifier(**params)`.

    Decisión del equipo: 100 árboles fijos, sin early stopping, en CPU y GPU, para que
    ambos dispositivos hagan exactamente el mismo trabajo al medir el speedup. Por eso
    `early_stopping=False` por defecto; `True` (agrega `early_stopping_rounds`, hay que
    pasar `eval_set` al `fit`) solo se usa como referencia de métricas en
    `referencia_cpu/linea_base_xgboost.py`.
    """
    params = dict(PARAMETROS_XGBOOST_BASE)
    params.update(
        learning_rate=learning_rate,
        max_leaves=max_leaves,
        scale_pos_weight=scale_pos_weight,
        device=device,
        n_jobs=n_jobs,
    )
    if early_stopping:
        params["early_stopping_rounds"] = RONDAS_SIN_MEJORA_ES
    return params


def combinaciones(rejilla: dict) -> list[dict]:
    """Producto cartesiano de una rejilla, en el orden de GridSearchCV."""
    claves = sorted(rejilla)
    return [dict(zip(claves, valores)) for valores in product(*(rejilla[c] for c in claves))]
