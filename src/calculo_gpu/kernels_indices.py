"""K1: índices por píxel con Numba CUDA. Responsable: Nadissa.

Recibe el cubo ya leído en memoria (`preparacion_cpu.lectura.leer_cubo`); no lee
archivos. La matemática es `src/indices_pixel.pixel`, compilada como función de
dispositivo, así que es la misma que la versión CPU multinúcleo.

Distribución del trabajo
------------------------
* Un hilo por píxel y fecha.
* Bloques 2D de 16 x 16 = 256 hilos; `threadIdx.x` recorre las columnas, de modo que
  hilos consecutivos leen direcciones consecutivas de cada banda (acceso coalescido,
  porque el cubo es C-contiguo con la columna como eje más rápido).
* Grid 3D: (ceil(ancho/16), ceil(alto/16), n_fechas). `blockIdx.z` es la fecha.
* Sin ramas de control salvo el chequeo de bordes: las condiciones de la máscara de
  agua se combinan con operaciones booleanas (ver `indices_pixel`).

Modos
-----
"simple"  : una copia CPU->GPU del cubo completo, un lanzamiento, una copia de vuelta.
            Tiempos separados de cada fase con eventos de CUDA.
"streams" : el cubo se parte en lotes de fechas; cada lote usa su propio stream
            (copia de ida, kernel, copia de vuelta), así la copia de un lote se
            solapa con el cálculo de otro. Requiere memoria del host bloqueada
            (pinned); `cuda.pinned` la bloquea durante la llamada.

Salidas (mismo contrato que `referencia_cpu.indices_multinucleo`):
    capas    float32 (fechas, 5, alto, ancho): ndvi, ndwi, ndci, fai, chla
    mascaras uint8   (fechas, 3, alto, ancho): agua, validos, y_alta
"""

from __future__ import annotations

import math
import time
from contextlib import ExitStack

import numpy as np
from numba import cuda

from src.indices_pixel import CAPAS_FLOAT, CAPAS_MASCARA, pixel

BLOQUE = (16, 16)
MODOS = ("simple", "streams")

_pixel_gpu = cuda.jit(device=True)(pixel)


@cuda.jit
def k1_indices(cubo, scl, capas, mascaras):
    col, fila = cuda.grid(2)
    f = cuda.blockIdx.z
    if fila < cubo.shape[2] and col < cubo.shape[3]:
        r = _pixel_gpu(
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


def configuracion(n_fechas: int, alto: int, ancho: int) -> tuple[tuple, tuple]:
    """(grid, bloque) para un lanzamiento sobre `n_fechas` fechas."""
    grid = (math.ceil(ancho / BLOQUE[0]), math.ceil(alto / BLOQUE[1]), n_fechas)
    return grid, BLOQUE


def _salidas(n_fechas: int, alto: int, ancho: int):
    capas = np.empty((n_fechas, len(CAPAS_FLOAT), alto, ancho), dtype=np.float32)
    mascaras = np.empty((n_fechas, len(CAPAS_MASCARA), alto, ancho), dtype=np.uint8)
    return capas, mascaras


def _simple(cubo, scl):
    n_fechas, _, alto, ancho = cubo.shape
    capas, mascaras = _salidas(n_fechas, alto, ancho)
    ev = [cuda.event() for _ in range(4)]

    ev[0].record()
    d_cubo = cuda.to_device(cubo)
    d_scl = cuda.to_device(scl)
    d_capas = cuda.device_array(capas.shape, dtype=np.float32)
    d_mascaras = cuda.device_array(mascaras.shape, dtype=np.uint8)
    ev[1].record()
    grid, bloque = configuracion(n_fechas, alto, ancho)
    k1_indices[grid, bloque](d_cubo, d_scl, d_capas, d_mascaras)
    ev[2].record()
    d_capas.copy_to_host(capas)
    d_mascaras.copy_to_host(mascaras)
    ev[3].record()
    ev[3].synchronize()

    tiempos = {
        "copia_cpu_gpu_s": ev[0].elapsed_time(ev[1]) / 1000,
        "kernel_s": ev[1].elapsed_time(ev[2]) / 1000,
        "copia_gpu_cpu_s": ev[2].elapsed_time(ev[3]) / 1000,
    }
    return capas, mascaras, tiempos


def _streams(cubo, scl, fechas_por_lote: int):
    n_fechas, _, alto, ancho = cubo.shape
    capas, mascaras = _salidas(n_fechas, alto, ancho)
    lotes = [(i, min(i + fechas_por_lote, n_fechas)) for i in range(0, n_fechas, fechas_por_lote)]

    inicio = time.perf_counter()
    with ExitStack() as pila:
        for arreglo in (cubo, scl, capas, mascaras):
            pila.enter_context(cuda.pinned(arreglo))
        bloqueo_s = time.perf_counter() - inicio
        ev_ini, ev_fin = cuda.event(), cuda.event()
        ev_ini.record()
        streams = []
        for a, b in lotes:
            s = cuda.stream()
            d_cubo = cuda.to_device(cubo[a:b], stream=s)
            d_scl = cuda.to_device(scl[a:b], stream=s)
            d_capas = cuda.device_array((b - a, len(CAPAS_FLOAT), alto, ancho), np.float32, stream=s)
            d_masc = cuda.device_array((b - a, len(CAPAS_MASCARA), alto, ancho), np.uint8, stream=s)
            grid, bloque = configuracion(b - a, alto, ancho)
            k1_indices[grid, bloque, s](d_cubo, d_scl, d_capas, d_masc)
            d_capas.copy_to_host(capas[a:b], stream=s)
            d_masc.copy_to_host(mascaras[a:b], stream=s)
            streams.append((s, d_cubo, d_scl, d_capas, d_masc))  # mantener vivos hasta sincronizar
        for s, *_ in streams:
            s.synchronize()
        ev_fin.record()
        ev_fin.synchronize()
        gpu_s = ev_ini.elapsed_time(ev_fin) / 1000

    tiempos = {
        "bloqueo_memoria_s": bloqueo_s,
        "gpu_solapado_s": gpu_s,
        "lotes": len(lotes),
    }
    return capas, mascaras, tiempos


def calcular_indices_gpu(
    cubo: np.ndarray, scl: np.ndarray, modo: str = "simple", fechas_por_lote: int = 2
) -> tuple[np.ndarray, np.ndarray, dict]:
    """K1 sobre el cubo completo de un lago.

    Returns:
        (capas, mascaras, tiempos). `tiempos` trae las fases medidas con eventos de
        CUDA (segundos): en "simple" copia CPU->GPU, kernel y copia GPU->CPU; en
        "streams" el tiempo total solapado y el de bloquear la memoria del host.
    """
    if modo not in MODOS:
        raise ValueError(f"modo debe ser uno de {MODOS}, no {modo!r}")
    assert cubo.dtype == np.float32 and cubo.flags["C_CONTIGUOUS"]
    assert scl.dtype == np.uint8 and scl.flags["C_CONTIGUOUS"]
    if modo == "simple":
        return _simple(cubo, scl)
    return _streams(cubo, scl, fechas_por_lote)
