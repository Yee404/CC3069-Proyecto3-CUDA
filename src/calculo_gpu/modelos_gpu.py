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

Cómo agregar la Regresión Logística propia (T4):
    1. Implementar `RegresionLogisticaGPU` en `src/calculo_gpu/regresion_logistica.py`.
    2. Escribir aquí la clase `RegresionLogisticaGPU_Modelo` con el mismo contrato
       (`nombre = "regresion_logistica"`, `rejilla = REJILLA_LOGISTICA`).
    3. Registrarla en `_FABRICAS` (abajo) y quitar el aviso de `crear_modelo`.
"""

from __future__ import annotations

import numpy as np

from src.calculo_gpu.xgboost_gpu import entrenar_xgboost_gpu, probabilidad_xgboost_gpu
from src.parametros_modelos import REJILLA_XGBOOST, SEMILLA

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


class XGBoostGPU(_ModeloGPU):
    """XGBoost `hist` con device="cuda": 100 árboles fijos, sin early stopping."""

    nombre = "xgboost"
    rejilla = REJILLA_XGBOOST

    def _calentar(self) -> None:
        """3 ajustes + predicciones con datos sintéticos del tamaño real (sin archivos).

        Compila los kernels de XGBoost, reserva memoria y sube la frecuencia de la GPU.
        Los datos tienen señal (no son ruido puro) para que los árboles lleguen a 31 hojas.
        """
        rng = np.random.default_rng(SEMILLA)
        X = rng.standard_normal((CALENTAMIENTO_FILAS, N_PREDICTORES), dtype=np.float32)
        ruido = rng.standard_normal(CALENTAMIENTO_FILAS, dtype=np.float32)
        puntaje = X[:, 0] + 0.5 * X[:, 1] + 0.3 * ruido
        y = (puntaje > np.quantile(puntaje, 0.988)).astype(np.int8)  # ~1.2 % de positivos
        params = {"learning_rate": 0.1, "max_leaves": 31}
        for _ in range(CALENTAMIENTO_AJUSTES):
            modelo = entrenar_xgboost_gpu(X, y, params)
            probabilidad_xgboost_gpu(modelo, X[:CALENTAMIENTO_PRUEBA])

    def ajustar(self, X, y, params: dict):
        # X e y se copian a la GPU aquí dentro (tiempo medido) salvo que ya estén en ella.
        return entrenar_xgboost_gpu(X, y, params)

    def probabilidad(self, modelo, X) -> np.ndarray:
        return probabilidad_xgboost_gpu(modelo, X)


# Espacio para T4: agregar aquí {"regresion_logistica": RegresionLogisticaGPU_Modelo}.
_FABRICAS = {"xgboost": XGBoostGPU}


def crear_modelo(nombre: str, variante: str = "gpu", calentar: bool = True) -> _ModeloGPU:
    """`crear_modelo("xgboost", "gpu")`. Misma firma que la versión CPU (+ `calentar`).

    Con `calentar=True` (por defecto) la GPU se calienta al crear el modelo; el protocolo
    llama `crear_modelo(nombre, variante)` y por tanto lo recibe sin cambios.
    """
    if nombre == "regresion_logistica" and nombre not in _FABRICAS:
        raise NotImplementedError(
            "La Regresión Logística en GPU (T4) aún no está registrada en modelos_gpu.py")
    if nombre not in _FABRICAS:
        raise ValueError(f"Modelo desconocido: {nombre!r}; use {sorted(_FABRICAS)}")
    return _FABRICAS[nombre](variante, calentar=calentar)
