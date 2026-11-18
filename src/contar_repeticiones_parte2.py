"""Cuenta las repeticiones reales de un ajuste de la Parte II (para la tabla de operaciones
repetitivas, paso 1 del enunciado). María José Yee.

Con el entrenamiento de `ej4_busqueda` (el ajuste representativo del protocolo) ajusta una vez
cada combinación de la rejilla y registra lo que el protocolo no guarda:

    Regresión Logística (C = 0.1, 1, 10), en CPU (scikit-learn, lbfgs) y en GPU (L-BFGS propio):
        iteraciones del optimizador, evaluaciones de f y su gradiente (solo GPU) y motivo de parada.
    XGBoost (4 combinaciones de la rejilla), en CPU y en GPU:
        árboles, hojas totales, hojas por árbol (media, mínimo y máximo) y profundidad (media y
        máxima), contadas en el volcado JSON de cada árbol.

No mide tiempos (eso lo hace `protocolo_parte2.py`) ni modifica ningún resultado existente.

Uso (desde la raíz del repositorio):

    .venv\\Scripts\\python -m src.contar_repeticiones_parte2

Salida: resultados/analisis/parte2_repeticiones_ej4.csv

Este script lee archivos, por eso vive fuera de `calculo_gpu/`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import RESULTADOS
from src.calculo_gpu.modelos_gpu import crear_modelo as crear_modelo_gpu
from src.parametros_modelos import combinaciones, REJILLA_LOGISTICA, REJILLA_XGBOOST
from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones
from src.referencia_cpu.modelos_cpu import crear_modelo as crear_modelo_cpu


def _forma_arbol(nodo: dict) -> tuple[int, int]:
    """(hojas, profundidad) de un árbol del volcado JSON de XGBoost."""
    if "children" not in nodo:
        return 1, 0
    hijos = [_forma_arbol(h) for h in nodo["children"]]
    return sum(h[0] for h in hijos), 1 + max(h[1] for h in hijos)


def _resumen_arboles(booster) -> dict:
    formas = [_forma_arbol(json.loads(arbol)) for arbol in booster.get_dump(dump_format="json")]
    hojas = np.array([f[0] for f in formas])
    profundidad = np.array([f[1] for f in formas])
    return {
        "arboles": len(formas), "hojas_total": int(hojas.sum()),
        "hojas_por_arbol_media": float(hojas.mean()), "hojas_min": int(hojas.min()),
        "hojas_max": int(hojas.max()), "profundidad_media": float(profundidad.mean()),
        "profundidad_max": int(profundidad.max()),
    }


def main() -> None:
    X, y = cargar_matrices()
    e4 = cargar_particiones()["ej4_busqueda"]
    X_ent, y_ent = X[e4.entrenamiento], y[e4.entrenamiento]
    base = {"n_entrenamiento": len(y_ent), "positivos": int(np.sum(y_ent == 1))}
    print(f"ej4_busqueda: {len(y_ent)} filas de entrenamiento, {base['positivos']} positivos")
    filas = []

    print("\nRegresión Logística (tol=1e-4, max_iter=1000)")
    cpu = crear_modelo_cpu("regresion_logistica", "cpu_secuencial")
    gpu = crear_modelo_gpu("regresion_logistica", "gpu")
    for params in combinaciones(REJILLA_LOGISTICA):
        pipe = cpu.ajustar(X_ent, y_ent, params)
        filas.append({**base, "modelo": "regresion_logistica", "version": "CPU (scikit-learn)",
                      "hiperparametros": json.dumps(params),
                      "iteraciones": int(pipe[-1].n_iter_[0])})
        m = gpu.ajustar(X_ent, y_ent, params)
        filas.append({**base, "modelo": "regresion_logistica", "version": "GPU (L-BFGS propio)",
                      "hiperparametros": json.dumps(params), "iteraciones": int(m.iteraciones),
                      "evaluaciones": int(m.evaluaciones), "motivo_parada": m.motivo_parada})
        print(f"  C={params['C']:<5} CPU {filas[-2]['iteraciones']:4d} iter | "
              f"GPU {m.iteraciones:4d} iter, {m.evaluaciones:4d} evaluaciones ({m.motivo_parada})")

    print("\nXGBoost (100 árboles fijos)")
    cpu = crear_modelo_cpu("xgboost", "cpu_multinucleo")  # los mismos árboles que el secuencial, más rápido
    gpu = crear_modelo_gpu("xgboost", "gpu")
    for params in combinaciones(REJILLA_XGBOOST):
        for version, modelo in (("CPU", cpu), ("GPU", gpu)):
            r = _resumen_arboles(modelo.ajustar(X_ent, y_ent, params).get_booster())
            filas.append({**base, "modelo": "xgboost", "version": version,
                          "hiperparametros": json.dumps(params), **r})
            print(f"  {json.dumps(params):42s} {version}: {r['arboles']} árboles, {r['hojas_total']} hojas "
                  f"({r['hojas_por_arbol_media']:.1f} por árbol, de {r['hojas_min']} a {r['hojas_max']}), "
                  f"profundidad media {r['profundidad_media']:.1f}, máxima {r['profundidad_max']}")

    carpeta = RESULTADOS / "analisis"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "parte2_repeticiones_ej4.csv"
    pd.DataFrame(filas).to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta.relative_to(RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
