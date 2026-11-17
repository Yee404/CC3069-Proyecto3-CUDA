"""XGBoost en GPU (device="cuda"). Responsable: María José Yee.

Mismos hiperparámetros que la versión CPU: salen de `src/parametros_modelos.py` y entre CPU
y GPU solo cambian `device` y `n_jobs` (aquí `n_jobs=1`: el CPU no ayuda al entrenamiento).

Contrato (este módulo NO lee ni escribe archivos; recibe arreglos ya preparados):

    entrenar_xgboost_gpu(X, y, params)  -> XGBClassifier ajustado en la GPU
    probabilidad_xgboost_gpu(modelo, X) -> ndarray NumPy (n,) en el host con P(y = 1)
    shap_gpu(modelo, X)                 -> ESQUELETO (opcional; se hace después de validar T3/T4)

`X` e `y` pueden ser:
    * arreglos NumPy en el host (float32 C-contiguo / int8): se copian a la GPU en cada llamada,
      y esa copia queda dentro del tiempo medido (decisión del equipo);
    * arreglos CuPy ya residentes en la GPU: se usan tal cual, sin copiarlos.

Datos en la GPU: `XGBClassifier.fit` construye internamente un `xgboost.QuantileDMatrix`
(formato ELLPACK comprimido) cuando el método es `hist`, así que con arreglos CuPy el
entrenamiento no pasa por el host. Se usa el mismo envoltorio (`XGBClassifier`) que en CPU
para que los hiperparámetros se traduzcan exactamente igual en ambos dispositivos.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import xgboost as xgb

from src.parametros_modelos import parametros_xgboost, peso_positivos


def _esta_en_gpu(arreglo) -> bool:
    return isinstance(arreglo, cp.ndarray)


def _a_gpu(arreglo):
    """Devuelve el arreglo en la GPU, C-contiguo. Si ya está allí no se copia."""
    if _esta_en_gpu(arreglo):
        return cp.ascontiguousarray(arreglo)  # no copia si ya es contiguo
    return cp.asarray(np.ascontiguousarray(arreglo))  # copia CPU -> GPU


def _sincronizar() -> None:
    """Espera a que la GPU termine (el cronómetro del protocolo mide hasta aquí)."""
    cp.cuda.Device().synchronize()


def entrenar_xgboost_gpu(X, y, params: dict) -> xgb.XGBClassifier:
    """Ajusta un XGBClassifier con device="cuda".

    `params` trae solo la parte de la rejilla (`learning_rate`, `max_leaves`). El
    `scale_pos_weight` se calcula en CADA ajuste con las etiquetas de ese entrenamiento,
    igual que en CPU.
    """
    # Las etiquetas pesan 1 byte por fila: traerlas al host para contar positivos es barato.
    y_host = cp.asnumpy(y) if _esta_en_gpu(y) else y
    parametros = parametros_xgboost(
        **params, scale_pos_weight=peso_positivos(y_host), device="cuda", n_jobs=1
    )
    X_gpu, y_gpu = _a_gpu(X), _a_gpu(y)
    modelo = xgb.XGBClassifier(**parametros)
    modelo.fit(X_gpu, y_gpu, verbose=False)
    _sincronizar()
    return modelo


def probabilidad_xgboost_gpu(modelo: xgb.XGBClassifier, X) -> np.ndarray:
    """P(y = 1) como arreglo NumPy del host (la copia GPU -> CPU se hace aquí dentro).

    Con X en la GPU la predicción es "en el sitio" (`inplace_predict`): evita el aviso de
    dispositivos distintos y la ruta lenta por DMatrix.
    """
    proba = modelo.predict_proba(_a_gpu(X))[:, 1]
    _sincronizar()
    return cp.asnumpy(proba)  # acepta también un ndarray de NumPy


def shap_gpu(modelo: xgb.XGBClassifier, X):
    """ESQUELETO (opcional). SHAP en GPU con GPUTreeShap de XGBoost.

    Idea para cuando se haga (después de validar T3 y T4):
        contribuciones = modelo.get_booster().predict(
            xgb.DMatrix(_a_gpu(X)), pred_contribs=True)   # (n, 13 + 1): la última es el sesgo
    y compararlo contra TreeSHAP en CPU sobre la misma submuestra (PLAN.md §6.2).
    """
    raise NotImplementedError("SHAP en GPU: pendiente (opcional, PLAN.md §6.2).")
