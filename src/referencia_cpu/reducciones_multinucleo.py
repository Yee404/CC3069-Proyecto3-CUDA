"""K2 en CPU con Numba: máscara fija del lago y estadísticas por fecha.

Misma lógica que la versión GPU (`calculo_gpu/reducciones.py`) y mismas salidas
(sumas y conteos de `estadisticas.CANTIDADES`, máximo, mediana y p90 de clorofila-a),
para comparar contra la versión secuencial del laboratorio (`procesar.resumen`).

    "1 nucleo"    : njit
    "multinucleo" : njit(parallel=True); la máscara reparte filas entre núcleos y las
                    estadísticas reparten fechas (11 por lago, una por núcleo)

Sin cache=True por la misma razón que en `indices_multinucleo.py`.

Responsable: Nadissa.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from src.estadisticas import FRACCION_AGUA_LAGO, N_BINS, Q, UMBRAL_CHLA_ALTO, UMBRAL_FAI

NDVI, NDWI, NDCI, FAI, CHLA = 0, 1, 2, 3, 4
AGUA, VALIDOS = 0, 1


def _mascara_lago(mascaras, lago):
    n_fechas, _, alto, ancho = mascaras.shape
    for fila in prange(alto):
        for col in range(ancho):
            cuenta = 0
            for f in range(n_fechas):
                cuenta += mascaras[f, AGUA, fila, col]
            lago[fila, col] = (np.float32(cuenta) / np.float32(n_fechas)) >= FRACCION_AGUA_LAGO


def _estadisticas(capas, mascaras, lago, acum, maximo, medianas, p90):
    n_fechas, _, alto, ancho = capas.shape
    for f in prange(n_fechas):
        valores = np.empty(alto * ancho, dtype=np.float32)
        a = np.zeros(Q, dtype=np.float64)
        mx = np.float32(-np.inf)
        k = 0
        for fila in range(alto):
            for col in range(ancho):
                if lago[fila, col] == 0 or mascaras[f, VALIDOS, fila, col] == 0:
                    continue
                a[0] += 1.0
                ndci = capas[f, NDCI, fila, col]
                chla = capas[f, CHLA, fila, col]
                ndvi = capas[f, NDVI, fila, col]
                ndwi = capas[f, NDWI, fila, col]
                fai = capas[f, FAI, fila, col]
                if not math.isnan(ndci):
                    a[1] += ndci
                    a[2] += 1.0
                if not math.isnan(chla):
                    a[3] += chla
                    a[4] += 1.0
                    if chla >= UMBRAL_CHLA_ALTO:
                        a[9] += 1.0
                    if chla > mx:
                        mx = chla
                    valores[k] = chla
                    k += 1
                if not math.isnan(ndvi):
                    a[5] += ndvi
                    a[6] += 1.0
                if not math.isnan(ndwi):
                    a[7] += ndwi
                    a[8] += 1.0
                if not math.isnan(fai):
                    a[10] += 1.0
                    if fai > UMBRAL_FAI:
                        a[11] += 1.0
        for q in range(Q):
            acum[f, q] = a[q]
        maximo[f] = mx
        if k > 0:
            medianas[f] = np.median(valores[:k])
            p90[f] = np.percentile(valores[:k], 90)
        else:
            medianas[f] = np.nan
            p90[f] = np.nan


_mascara_1 = njit(error_model="numpy")(_mascara_lago)
_mascara_n = njit(parallel=True, error_model="numpy")(_mascara_lago)
_estad_1 = njit(error_model="numpy")(_estadisticas)
_estad_n = njit(parallel=True, error_model="numpy")(_estadisticas)


def estadisticas_cpu(capas: np.ndarray, mascaras: np.ndarray, paralelo: bool = True) -> dict:
    """Máscara del lago y K2 en CPU sobre las salidas de K1.

    Returns:
        dict con `lago` (uint8), `acum` (fechas x Q), `maximo`, `medianas`, `p90`.
    """
    n_fechas, _, alto, ancho = capas.shape
    lago = np.empty((alto, ancho), dtype=np.uint8)
    acum = np.empty((n_fechas, Q), dtype=np.float64)
    maximo = np.empty(n_fechas, dtype=np.float32)
    medianas = np.empty(n_fechas, dtype=np.float64)
    p90 = np.empty(n_fechas, dtype=np.float64)
    (_mascara_n if paralelo else _mascara_1)(mascaras, lago)
    (_estad_n if paralelo else _estad_1)(capas, mascaras, lago, acum, maximo, medianas, p90)
    return {"lago": lago, "acum": acum, "maximo": maximo, "medianas": medianas, "p90": p90}


def mascara_lago_cpu(mascaras: np.ndarray, paralelo: bool = True) -> np.ndarray:
    """Máscara fija del lago (uint8) a partir de la capa `agua` de K1."""
    _, _, alto, ancho = mascaras.shape
    lago = np.empty((alto, ancho), dtype=np.uint8)
    (_mascara_n if paralelo else _mascara_1)(mascaras, lago)
    return lago


# --------------------------------------------------------------------------
# K3: histogramas de clorofila-a por fecha (misma elección de bin que la GPU)
# --------------------------------------------------------------------------
def _histogramas(capas, mascaras, lago, bordes, hist):
    n_fechas, _, alto, ancho = capas.shape
    inicio = bordes[0]
    fin = bordes[N_BINS]
    for f in prange(n_fechas):
        for fila in range(alto):
            for col in range(ancho):
                if lago[fila, col] == 0 or mascaras[f, VALIDOS, fila, col] == 0:
                    continue
                x = capas[f, CHLA, fila, col]
                if math.isnan(x):
                    continue
                if x < inicio:
                    x = inicio
                if x > fin:
                    x = fin
                b = int((x - inicio) / (fin - inicio) * N_BINS)
                if b > N_BINS - 1:
                    b = N_BINS - 1
                while b > 0 and x < bordes[b]:
                    b -= 1
                while b < N_BINS - 1 and x >= bordes[b + 1]:
                    b += 1
                hist[f, b] += 1


_hist_1 = njit(error_model="numpy")(_histogramas)
_hist_n = njit(parallel=True, error_model="numpy")(_histogramas)


def histogramas_cpu(capas, mascaras, lago, bordes: np.ndarray, paralelo: bool = True) -> np.ndarray:
    """K3 en CPU: (fechas, N_BINS) conteos. En paralelo, una fecha por núcleo."""
    hist = np.zeros((capas.shape[0], N_BINS), dtype=np.int64)
    (_hist_n if paralelo else _hist_1)(capas, mascaras, lago, bordes.astype(np.float32), hist)
    return hist

