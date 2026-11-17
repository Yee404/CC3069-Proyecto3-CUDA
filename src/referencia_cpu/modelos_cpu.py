"""Modelos de la Parte II en CPU (secuencial y multinúcleo). Responsable: María José Yee.

    Regresión Logística : StandardScaler + LogisticRegression (scikit-learn),
                          class_weight="balanced", rejilla C en {0.1, 1, 10}
    XGBoost             : tree_method="hist", device="cpu"

Cada modelo se presenta al orquestador (`src/protocolo_parte2.py`) con la MISMA interfaz que
tendrá la versión GPU, para que el protocolo, las particiones y las métricas sean idénticos:

    nombre               "regresion_logistica" | "xgboost"
    variante             "cpu_secuencial" | "cpu_multinucleo"   (la GPU será "gpu")
    version              etiqueta de `medicion/tiempos.py`: "CPU secuencial" | "CPU multinucleo"
    rejilla              rejilla de hiperparámetros (viene de `parametros_modelos.py`)
    ajustes_en_paralelo  True: el orquestador reparte ajustes independientes entre procesos
    sincronizar_gpu      False en CPU (True en la versión GPU)
    ajustar(X, y, params)        -> modelo ajustado   (X float32 C-contiguo, y int8)
    probabilidad(modelo, X)      -> ndarray (n,) con P(y = 1)

Dónde vive el paralelismo de la versión multinúcleo:
    * XGBoost: dentro de cada ajuste (`n_jobs=-1`, hilos de OpenMP). Secuencial: `n_jobs=1`.
    * Regresión Logística: `LogisticRegression` (lbfgs) no paraleliza un problema binario,
      así que el paralelismo está ENTRE ajustes independientes (cada combinación de C y cada
      pliegue), como haría `GridSearchCV(n_jobs=-1)`. El reparto lo hace el orquestador con
      joblib. La versión secuencial limita BLAS a 1 hilo con threadpoolctl para que sea
      secuencial de verdad.

Los hiperparámetros NO se repiten aquí: salen de `src/parametros_modelos.py`.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.parametros_modelos import (
    PARAMETROS_LOGISTICA,
    REJILLA_LOGISTICA,
    REJILLA_XGBOOST,
    parametros_xgboost,
    peso_positivos,
)

VARIANTES_CPU = ("cpu_secuencial", "cpu_multinucleo")
_ETIQUETA_VERSION = {"cpu_secuencial": "CPU secuencial", "cpu_multinucleo": "CPU multinucleo"}


class _ModeloCPU:
    """Base común: guarda la variante y la etiqueta de tiempos."""

    nombre: str = ""
    rejilla: dict = {}
    sincronizar_gpu = False  # en CPU no hay nada que sincronizar

    def __init__(self, variante: str) -> None:
        if variante not in VARIANTES_CPU:
            raise ValueError(f"Variante CPU desconocida: {variante!r}; use {VARIANTES_CPU}")
        self.variante = variante
        self.version = _ETIQUETA_VERSION[variante]
        self.multinucleo = variante == "cpu_multinucleo"

    # Contrato con el orquestador (se sobrescribe en cada modelo).
    ajustes_en_paralelo = False

    def ajustar(self, X: np.ndarray, y: np.ndarray, params: dict):  # pragma: no cover
        raise NotImplementedError

    def probabilidad(self, modelo, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class RegresionLogisticaCPU(_ModeloCPU):
    """StandardScaler + LogisticRegression(class_weight="balanced"), como el notebook 04."""

    nombre = "regresion_logistica"
    rejilla = REJILLA_LOGISTICA

    def __init__(self, variante: str) -> None:
        super().__init__(variante)
        # Multinúcleo: ajustes independientes en procesos distintos (ver docstring del módulo).
        self.ajustes_en_paralelo = self.multinucleo

    def _hilos_blas(self):
        # Secuencial = 1 hilo de BLAS; multinúcleo = sin límite (los procesos de joblib
        # ya reciben su propio límite para no sobresuscribir el CPU).
        return nullcontext() if self.multinucleo else threadpool_limits(limits=1)

    def ajustar(self, X: np.ndarray, y: np.ndarray, params: dict):
        # float64 INTERNO: con float32, lbfgs de scikit-learn da coeficientes que dependen de
        # los hilos de BLAS (secuencial != multinúcleo) y un PR-AUC ~0.006 menor; con float64
        # el resultado es idéntico con 1 hilo o con todos y cercano al laboratorio (que
        # trabajaba en float64). X sigue siendo float32 C-contiguo fuera de este método.
        with self._hilos_blas():
            modelo = make_pipeline(
                StandardScaler(),
                LogisticRegression(**PARAMETROS_LOGISTICA, **params),
            )
            return modelo.fit(X.astype(np.float64), y)

    def probabilidad(self, modelo, X: np.ndarray) -> np.ndarray:
        with self._hilos_blas():
            return modelo.predict_proba(X.astype(np.float64))[:, 1]


class XGBoostCPU(_ModeloCPU):
    """XGBoost `hist` en CPU: 100 árboles fijos, sin early stopping (decisión del equipo)."""

    nombre = "xgboost"
    rejilla = REJILLA_XGBOOST
    ajustes_en_paralelo = False  # el paralelismo está dentro de cada ajuste (n_jobs)

    def __init__(self, variante: str) -> None:
        super().__init__(variante)
        self.n_jobs = -1 if self.multinucleo else 1

    def ajustar(self, X: np.ndarray, y: np.ndarray, params: dict):
        # scale_pos_weight se calcula en CADA ajuste con las etiquetas de ese entrenamiento.
        parametros = parametros_xgboost(
            **params, scale_pos_weight=peso_positivos(y), device="cpu", n_jobs=self.n_jobs
        )
        return xgb.XGBClassifier(**parametros).fit(X, y, verbose=False)

    def probabilidad(self, modelo, X: np.ndarray) -> np.ndarray:
        return modelo.predict_proba(X)[:, 1]


_FABRICAS = {"regresion_logistica": RegresionLogisticaCPU, "xgboost": XGBoostCPU}


def crear_modelo(nombre: str, variante: str) -> _ModeloCPU:
    """`crear_modelo("xgboost", "cpu_multinucleo")`. La versión GPU tendrá la misma firma."""
    if nombre not in _FABRICAS:
        raise ValueError(f"Modelo desconocido: {nombre!r}; use {sorted(_FABRICAS)}")
    return _FABRICAS[nombre](variante)
