"""Fase 1: reemplazo de HistGradientBoostingClassifier por XGBoost en CPU.

Repite el Ej. 4 del notebook 04 (búsqueda de hiperparámetros con CV interna de 3
pliegues sobre el 70 % y evaluación en el 30 % de prueba) con las particiones fijas
de `preparacion_cpu.dataset`, para tres modelos:

    HGB           HistGradientBoostingClassifier del laboratorio, rejilla original
                  (reproduce el 0.9706 y dice cuántos árboles usó: `n_iter_`)
    XGB 100       XGBoost CPU, 100 árboles fijos, sin early stopping
                  (configuración elegida por el equipo para CPU y GPU)
    XGB ES        XGBoost CPU con early stopping como HGB (10 % de validación
                  estratificada dentro de cada ajuste, 10 rondas sin mejora); solo
                  como referencia de cuánto cambian las métricas al fijar 100 árboles

Referencia del laboratorio: PR-AUC de validación interna 0.9706 y en prueba 0.9660
(precision 0.7636, recall 0.9877 con umbral 0.5).

Este script valida la equivalencia; NO es la medición de tiempos (esa la hace
`referencia_cpu/modelos_cpu.py` con `medicion/tiempos.py`). Usa todos los núcleos.

    .venv\\Scripts\\python -m src.referencia_cpu.linea_base_xgboost
    .venv\\Scripts\\python -m src.referencia_cpu.linea_base_xgboost --sin-hgb

Salidas en resultados/validacion/:
    linea_base_busqueda.csv   PR-AUC por pliegue y combinación, árboles usados
    linea_base_prueba.csv     métricas en prueba (umbral 0.5 y umbral óptimo F2)
    linea_base_xgboost.json   mejores hiperparámetros y n_iter_ de HGB

Responsable: Javier Linares.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src import RESULTADOS
from src.evaluacion import metricas, umbral_optimo
from src.parametros_modelos import (
    FRACCION_VALIDACION_ES,
    REJILLA_XGBOOST,
    SEMILLA,
    combinaciones,
    parametros_xgboost,
    peso_positivos,
)
from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones

REJILLA_HGB = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63]}

LAB = {
    "pr_auc_validacion": 0.9706,
    "mejores": {"learning_rate": 0.1, "max_leaf_nodes": 31},
    "prueba": {"precision": 0.7636, "recall": 0.9877, "f2": 0.9329, "roc_auc": 0.9996,
               "pr_auc": 0.9660},
}


# --------------------------------------------------------------------------
# Ajuste de un modelo
# --------------------------------------------------------------------------
def ajustar_hgb(X, y, params: dict) -> tuple[HistGradientBoostingClassifier, int]:
    modelo = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=SEMILLA, **params
    ).fit(X, y)
    return modelo, int(modelo.n_iter_)


def ajustar_xgb(X, y, params: dict, early_stopping: bool, n_jobs: int = -1):
    """XGBoost CPU. Con early stopping separa el 10 % estratificado como validación."""
    if early_stopping:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=FRACCION_VALIDACION_ES, stratify=y, random_state=SEMILLA
        )
    else:
        X_fit, y_fit = X, y
    spw = peso_positivos(y_fit)
    modelo = xgb.XGBClassifier(
        **parametros_xgboost(**params, scale_pos_weight=spw, device="cpu", n_jobs=n_jobs,
                             early_stopping=early_stopping)
    )
    if early_stopping:
        # El logloss de validación se pondera igual que el entrenamiento (como HGB).
        peso_val = np.where(y_val == 1, spw, 1.0)
        modelo.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                   sample_weight_eval_set=[peso_val], verbose=False)
        arboles = int(modelo.best_iteration) + 1
    else:
        modelo.fit(X_fit, y_fit, verbose=False)
        arboles = modelo.get_booster().num_boosted_rounds()
    return modelo, arboles


MODELOS = {
    "HGB (laboratorio)": (REJILLA_HGB, lambda X, y, p: ajustar_hgb(X, y, p)),
    "XGBoost CPU 100 arboles": (REJILLA_XGBOOST, lambda X, y, p: ajustar_xgb(X, y, p, False)),
    "XGBoost CPU early stopping": (REJILLA_XGBOOST, lambda X, y, p: ajustar_xgb(X, y, p, True)),
}


# --------------------------------------------------------------------------
# Búsqueda (equivalente a GridSearchCV(scoring="average_precision", refit=True))
# --------------------------------------------------------------------------
def buscar(nombre: str, X, y, experimento) -> tuple[dict, list[dict], object, int]:
    rejilla, ajustar = MODELOS[nombre]
    filas = []
    for params in combinaciones(rejilla):
        puntajes, arboles = [], []
        for ent, val in experimento.pliegues:
            modelo, n_arboles = ajustar(X[ent], y[ent], params)
            puntajes.append(average_precision_score(y[val], modelo.predict_proba(X[val])[:, 1]))
            arboles.append(n_arboles)
        filas.append({
            "modelo": nombre, **{k: params[k] for k in sorted(params)},
            **{f"pr_auc_pliegue{k}": p for k, p in enumerate(puntajes)},
            "pr_auc_media": float(np.mean(puntajes)),
            "arboles_por_pliegue": ";".join(map(str, arboles)),
        })
        print(f"  {nombre:28s} {params}  PR-AUC(cv) = {np.mean(puntajes):.4f}  árboles {arboles}")

    # GridSearchCV se queda con la primera combinación de mayor puntaje.
    mejor = max(filas, key=lambda f: f["pr_auc_media"])
    mejores = {k: mejor[k] for k in rejilla}
    modelo, n_arboles = ajustar(X[experimento.entrenamiento], y[experimento.entrenamiento], mejores)
    return mejores, filas, modelo, n_arboles


def main() -> None:
    parser = argparse.ArgumentParser(description="Línea base XGBoost CPU (Fase 1)")
    parser.add_argument("--sin-hgb", action="store_true", help="omitir la búsqueda con HGB")
    args = parser.parse_args()

    X, y = cargar_matrices()
    e4 = cargar_particiones()["ej4_busqueda"]
    X_pru, y_pru = X[e4.prueba], y[e4.prueba]

    busqueda, prueba, resumen = [], [], {"laboratorio": LAB, "modelos": {}}
    for nombre in MODELOS:
        if args.sin_hgb and nombre.startswith("HGB"):
            continue
        print(f"\n=== {nombre} ===")
        mejores, filas, modelo, n_arboles = buscar(nombre, X, y, e4)
        busqueda += filas
        proba = modelo.predict_proba(X_pru)[:, 1]
        umbral, _ = umbral_optimo(y_pru, proba)
        extra = {"arboles_modelo_final": n_arboles, "hiperparametros": json.dumps(mejores)}
        prueba.append(metricas(nombre, y_pru, proba, 0.5, {**extra, "criterio_umbral": "0.5"}))
        prueba.append(metricas(nombre, y_pru, proba, umbral, {**extra, "criterio_umbral": "optimo F2"}))
        resumen["modelos"][nombre] = {
            "mejores_hiperparametros": mejores,
            "pr_auc_validacion": round(max(f["pr_auc_media"] for f in filas), 4),
            "pr_auc_prueba": round(prueba[-1]["pr_auc"], 4),
            "arboles_modelo_final": n_arboles,
        }
        print(f"  mejor {mejores} -> PR-AUC prueba {prueba[-1]['pr_auc']:.4f}, "
              f"árboles del modelo final: {n_arboles}")

    destino = RESULTADOS / "validacion"
    destino.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(busqueda).to_csv(destino / "linea_base_busqueda.csv", index=False)
    tabla = pd.DataFrame(prueba)
    tabla.to_csv(destino / "linea_base_prueba.csv", index=False)
    (destino / "linea_base_xgboost.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nComparación en prueba (umbral 0.5) contra el laboratorio:")
    vista = tabla[tabla["criterio_umbral"] == "0.5"].set_index("modelo")
    vista = vista[["precision", "recall", "f2", "roc_auc", "pr_auc", "arboles_modelo_final"]]
    print(vista.round(4).to_string())
    print(f"Laboratorio (HGB): {LAB['prueba']}")
    print("\nGuardado en resultados/validacion/: linea_base_busqueda.csv, "
          "linea_base_prueba.csv, linea_base_xgboost.json")


if __name__ == "__main__":
    main()
