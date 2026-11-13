"""K1 en CPU con Numba: un núcleo (`njit`) y multinúcleo (`njit(parallel=True)` + prange).

Misma matemática que la GPU (`src/indices_pixel.pixel`) y mismo contrato:

    entrada: cubo float32 (fechas, 9, alto, ancho) y scl uint8 (fechas, alto, ancho)
             de `preparacion_cpu.lectura.leer_cubo`
    salida:  capas float32 (fechas, 5, alto, ancho) en el orden de CAPAS_FLOAT
             mascaras uint8 (fechas, 3, alto, ancho) en el orden de CAPAS_MASCARA

Hay dos versiones compiladas del mismo bucle porque separan dos efectos que en el
informe conviene distinguir:

    "numba 1 nucleo"  : fusión de operaciones (cada píxel se calcula completo en
                        registros, recorriendo las bandas una sola vez) sin paralelismo
    "numba prange"    : fusión + reparto de filas entre todos los núcleos

Comparadas con `lab4/indices.py` (NumPy, que crea un arreglo temporal por operación),
la primera mide cuánto se gana solo por fusionar y la segunda cuánto agregan los núcleos.

Responsable: Nadissa.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

from src.indices_pixel import CAPAS_FLOAT, CAPAS_MASCARA, pixel

_pixel_cpu = njit(error_model="numpy", cache=True)(pixel)


def _bucle(cubo, scl, capas, mascaras):
    n_fechas, _, alto, ancho = cubo.shape
    for k in prange(n_fechas * alto):
        f = k // alto
        fila = k % alto
        for col in range(ancho):
            r = _pixel_cpu(
                cubo[f, 0, fila, col], cubo[f, 1, fila, col], cubo[f, 2, fila, col],
                cubo[f, 3, fila, col], cubo[f, 4, fila, col], cubo[f, 5, fila, col],
                cubo[f, 6, fila, col], cubo[f, 7, fila, col], cubo[f, 8, fila, col],
                scl[f, fila, col],
            )
            capas[f, 0, fila, col] = r[0]
            capas[f, 1, fila, col] = r[1]
            capas[f, 2, fila, col] = r[2]
            capas[f, 3, fila, col] = r[3]
            capas[f, 4, fila, col] = r[4]
            mascaras[f, 0, fila, col] = r[5]
            mascaras[f, 1, fila, col] = r[6]
            mascaras[f, 2, fila, col] = r[7]


# Sin parallel=True, prange se comporta como range: mismo código, un núcleo.
# Sin cache=True a propósito: Numba nombra el archivo de caché según la función
# original (`_bucle`), así que las dos versiones compartían el mismo archivo y la
# multinúcleo terminaba cargando el código compilado de la de 1 núcleo (medían
# lo mismo). La compilación ocurre en la corrida de calentamiento, que no se mide.
_bucle_1_nucleo = njit(error_model="numpy")(_bucle)
_bucle_prange = njit(parallel=True, error_model="numpy")(_bucle)


def salidas_vacias(cubo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_fechas, _, alto, ancho = cubo.shape
    capas = np.empty((n_fechas, len(CAPAS_FLOAT), alto, ancho), dtype=np.float32)
    mascaras = np.empty((n_fechas, len(CAPAS_MASCARA), alto, ancho), dtype=np.uint8)
    return capas, mascaras


def calcular_indices_cpu(
    cubo: np.ndarray, scl: np.ndarray, paralelo: bool = True, n_hilos: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """K1 sobre el cubo completo en CPU.

    Args:
        paralelo: True = prange en todos los núcleos; False = un núcleo.
        n_hilos: hilos de Numba para la versión paralela (None = todos).
    """
    assert cubo.dtype == np.float32 and cubo.flags["C_CONTIGUOUS"]
    assert scl.dtype == np.uint8 and scl.shape[0] == cubo.shape[0]
    capas, mascaras = salidas_vacias(cubo)
    if paralelo:
        anterior = numba.get_num_threads()
        if n_hilos:
            numba.set_num_threads(n_hilos)
        try:
            _bucle_prange(cubo, scl, capas, mascaras)
        finally:
            numba.set_num_threads(anterior)
    else:
        _bucle_1_nucleo(cubo, scl, capas, mascaras)
    return capas, mascaras
