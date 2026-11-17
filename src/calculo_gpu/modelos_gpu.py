"""Modelos de la Parte II en GPU, con la MISMA interfaz que `referencia_cpu/modelos_cpu.py`.

`src/protocolo_parte2.py` los usa sin cambios (`crear_modelo(nombre, "gpu")`):

    nombre               "regresion_logistica" | "xgboost"
    variante             "gpu"
    version              "GPU" (etiqueta de `medicion/tiempos.py`)
    rejilla              rejilla de hiperparámetros (de `parametros_modelos.py`)
    ajustes_en_paralelo  False: los ajustes se hacen uno tras otro; la GPU es la que paraleliza
    sincronizar_gpu      True: el cronómetro llama a cuda.synchronize() antes de parar
    ajustar(X, y, params)        -> modelo ajustado (X float32 C-contiguo, y int8)
    probabilidad(modelo, X)      -> ndarray NumPy (n,) con P(y = 1)

Este módulo no lee ni escribe archivos (regla de separación CPU/GPU).

Modelos:
    "regresion_logistica"  implementación propia en Numba CUDA (`regresion_logistica.py`, T4)
    "xgboost"              XGBoost device="cuda" (`xgboost_gpu.py`, T3)
"""

from __future__ import annotations

import numpy as np

from src.calculo_gpu.regresion_logistica import RegresionLogisticaGPU
from src.calculo_gpu.xgboost_gpu import entrenar_xgboost_gpu, probabilidad_xgboost_gpu
from src.parametros_modelos import (
    PARAMETROS_LOGISTICA,
    REJILLA_LOGISTICA,
    REJILLA_XGBOOST,
    SEMILLA,
)

VARIANTES_GPU = ("gpu",)

# Calentamiento de la GPU (se hace al crear el modelo, ANTES de cualquier cronómetro).
# Con un solo ajuste chico la GPU de la laptop no sube su frecuencia y el primer
# experimento medido sale ~2x más lento; 3 ajustes del tamaño real la dejan en régimen.
CALENTAMIENTO_AJUSTES = 3
CALENTAMIENTO_FILAS = 281_000   # tamaño de un entrenamiento de ej4_busqueda
CALENTAMIENTO_PRUEBA = 120_000  # tamaño de la partición de prueba
N_PREDICTORES = 13              # dataset_ml.PREDICTORES


class _ModeloGPU:
    """Base común: variante, etiqueta de tiempos y banderas del contrato."""

    nombre: str = ""
    rejilla: dict = {}
    sincronizar_gpu = True  # los kernels son asíncronos: hay que sincronizar al medir
    ajustes_en_paralelo = False  # nunca repartir ajustes de GPU entre procesos

    def __init__(self, variante: str, calentar: bool = True) -> None:
        if variante not in VARIANTES_GPU:
            raise ValueError(f"Variante GPU desconocida: {variante!r}; use {VARIANTES_GPU}")
        self.variante = variante
        self.version = "GPU"
        if calentar:
            self._calentar()  # descartado: ocurre antes de que el protocolo empiece a medir

    def _calentar(self) -> None:
        """Gancho de calentamiento; cada modelo lo sobrescribe (T4 hará lo suyo)."""

    def ajustar(self, X: np.ndarray, y: np.ndarray, params: dict):  # pragma: no cover
        raise NotImplementedError

    def probabilidad(self, modelo, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


def datos_calentamiento() -> tuple[np.ndarray, np.ndarray]:
    """Datos sintéticos del tamaño real (sin archivos), con señal y ~1.2 % de positivos."""
    rng = np.random.default_rng(SEMILLA)
    X = rng.standard_normal((CALENTAMIENTO_FILAS, N_PREDICTORES), dtype=np.float32)
    ruido = rng.standard_normal(CALENTAMIENTO_FILAS, dtype=np.float32)
    puntaje = X[:, 0] + 0.5 * X[:, 1] + 0.3 * ruido
    y = (puntaje > np.quantile(puntaje, 0.988)).astype(np.int8)
    return X, y


class RegresionLogisticaGPU_Modelo(_ModeloGPU):
    """Regresión Logística propia en CUDA, equivalente a StandardScaler + LogisticRegression.

    Mismos hiperparámetros que la versión CPU (`PARAMETROS_LOGISTICA`): class_weight
    "balanced" (fijo en la implementación) y `max_iter`; la rejilla solo cambia `C`.
    """

    nombre = "regresion_logistica"
    rejilla = REJILLA_LOGISTICA

    def __init__(self, variante: str, calentar: bool = True) -> None:
        if PARAMETROS_LOGISTICA.get("class_weight") != "balanced":
            raise ValueError("La versión GPU implementa solo class_weight='balanced'")
        self.max_iter = PARAMETROS_LOGISTICA["max_iter"]
        super().__init__(variante, calentar)

    def _calentar(self) -> None:
        """3 ajustes + predicciones del tamaño real: compila los kernels (JIT de Numba),
        llena el pool de memoria de CuPy y sube la frecuencia de la GPU."""
        X, y = datos_calentamiento()
        for _ in range(CALENTAMIENTO_AJUSTES):
            modelo = self.ajustar(X, y, {"C": 1.0})
            self.probabilidad(modelo, X[:CALENTAMIENTO_PRUEBA])

    def ajustar(self, X, y, params: dict) -> RegresionLogisticaGPU:
        # X e y se copian a la GPU aquí dentro (tiempo medido) salvo que ya estén en ella.
        return RegresionLogisticaGPU(C=params["C"], max_iter=self.max_iter).ajustar(X, y)

    def probabilidad(self, modelo: RegresionLogisticaGPU, X) -> np.ndarray:
        return modelo.probabilidad(X)


class XGBoostGPU(_ModeloGPU):
    """XGBoost `hist` con device="cuda": 100 árboles fijos, sin early stopping."""

    nombre = "xgboost"
    rejilla = REJILLA_XGBOOST

    def _calentar(self) -> None:
        """3 ajustes + predicciones con datos sintéticos del tamaño real (sin archivos).

        Compila los kernels de XGBoost, reserva memoria y sube la frecuencia de la GPU.
        Los datos tienen señal (no son ruido puro) para que los árboles lleguen a 31 hojas.
        """
        X, y = datos_calentamiento()
        params = {"learning_rate": 0.1, "max_leaves": 31}
        for _ in range(CALENTAMIENTO_AJUSTES):
            modelo = entrenar_xgboost_gpu(X, y, params)
            probabilidad_xgboost_gpu(modelo, X[:CALENTAMIENTO_PRUEBA])

    def ajustar(self, X, y, params: dict):
        # X e y se copian a la GPU aquí dentro (tiempo medido) salvo que ya estén en ella.
        return entrenar_xgboost_gpu(X, y, params)

    def probabilidad(self, modelo, X) -> np.ndarray:
        return probabilidad_xgboost_gpu(modelo, X)


_FABRICAS = {"regresion_logistica": RegresionLogisticaGPU_Modelo, "xgboost": XGBoostGPU}


def crear_modelo(nombre: str, variante: str = "gpu", calentar: bool = True) -> _ModeloGPU:
    """`crear_modelo("xgboost", "gpu")`. Misma firma que la versión CPU (+ `calentar`).

    Con `calentar=True` (por defecto) la GPU se calienta al crear el modelo; el protocolo
    llama `crear_modelo(nombre, variante)` y por tanto lo recibe sin cambios.
    """
    if nombre not in _FABRICAS:
        raise ValueError(f"Modelo desconocido: {nombre!r}; use {sorted(_FABRICAS)}")
    return _FABRICAS[nombre](variante, calentar=calentar)
