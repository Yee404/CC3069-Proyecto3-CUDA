"""K2: máscara fija del lago y estadísticas por fecha con Numba CUDA. Responsable: Nadissa.

Reproduce `procesar.construir_mascara_lago` y `procesar.resumen` del laboratorio sobre
las salidas de K1 **que ya están en la GPU** (`kernels_indices.k1_en_dispositivo`), así
las capas no viajan de vuelta al CPU: solo regresan las sumas, los conteos y los
valores de clorofila-a necesarios para la mediana y el percentil 90.

Kernels
-------
k_mascara_lago : un hilo por píxel recorre sus fechas y cuenta en cuántas es agua;
                 el píxel es lago si cuenta / fechas >= 0.6 (mismo cálculo en float32
                 que el laboratorio).
k2_reducciones : un hilo por píxel y fecha (mismo grid que K1). Cada hilo aporta sus
                 valores a 12 acumuladores (CANTIDADES) en memoria compartida; el bloque
                 los reduce en árbol (256 -> 1) y el hilo 0 hace una sola suma atómica por
                 acumulador en la memoria global. Las sumas son float64 (más exactas que
                 la suma float32 de NumPy; la diferencia se reporta en la validación).
                 El máximo de clorofila-a se reduce igual con `atomic.max`.
                 Además compacta los valores de clorofila-a usables en un búfer por
                 fecha: cada hilo toma un lugar dentro del bloque con un atómico en
                 memoria compartida y el bloque reserva su tramo del búfer global con un
                 solo atómico (en vez de un atómico global por píxel).

Mediana y percentil 90
----------------------
Necesitan ordenar. Se copian al CPU solo los valores compactados (≈ 1.2 MB por fecha
en Atitlán, contra ≈ 24 MB de las 5 capas) y se calculan con NumPy, que usa el mismo
método que el laboratorio (`estadisticas.orden_en_cpu`, con las fechas repartidas
entre hilos del CPU). Se descartó CuPy aquí para
no depender de que encuentre el CUDA Toolkit (NVRTC) en todas las máquinas.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float32, float64, int32

from src.calculo_gpu.kernels_indices import BLOQUE, configuracion
from src.estadisticas import FRACCION_AGUA_LAGO, N_BINS, Q, UMBRAL_CHLA_ALTO, UMBRAL_FAI

HILOS = BLOQUE[0] * BLOQUE[1]   # 256 hilos por bloque
MENOS_INF = np.float32(-np.inf)

# Índices de capas y máscaras de K1 (indices_pixel.CAPAS_FLOAT / CAPAS_MASCARA)
NDVI, NDWI, NDCI, FAI, CHLA = 0, 1, 2, 3, 4
VALIDOS = 1


@cuda.jit
def k_mascara_lago(mascaras, lago):
    col, fila = cuda.grid(2)
    if fila < lago.shape[0] and col < lago.shape[1]:
        n_fechas = mascaras.shape[0]
        cuenta = 0
        for f in range(n_fechas):
            cuenta += mascaras[f, 0, fila, col]   # capa "agua"
        lago[fila, col] = (float32(cuenta) / float32(n_fechas)) >= FRACCION_AGUA_LAGO


@cuda.jit
def k2_reducciones(capas, mascaras, lago, acum, maximo, valores, contador):
    col, fila = cuda.grid(2)
    f = cuda.blockIdx.z
    tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x

    sh = cuda.shared.array(shape=(Q, HILOS), dtype=float64)
    sh_max = cuda.shared.array(shape=HILOS, dtype=float32)
    sh_n = cuda.shared.array(shape=1, dtype=int32)
    sh_base = cuda.shared.array(shape=1, dtype=int32)

    for q in range(Q):
        sh[q, tid] = 0.0
    sh_max[tid] = MENOS_INF
    if tid == 0:
        sh_n[0] = 0
    cuda.syncthreads()

    pos = -1
    chla = float32(0.0)
    if fila < lago.shape[0] and col < lago.shape[1]:
        if (lago[fila, col] != 0) and (mascaras[f, VALIDOS, fila, col] != 0):
            sh[0, tid] = 1.0                                   # n_usable
            ndci = capas[f, NDCI, fila, col]
            chla = capas[f, CHLA, fila, col]
            ndvi = capas[f, NDVI, fila, col]
            ndwi = capas[f, NDWI, fila, col]
            fai = capas[f, FAI, fila, col]
            if not math.isnan(ndci):
                sh[1, tid] = ndci
                sh[2, tid] = 1.0
            if not math.isnan(chla):
                sh[3, tid] = chla
                sh[4, tid] = 1.0
                if chla >= UMBRAL_CHLA_ALTO:
                    sh[9, tid] = 1.0
                sh_max[tid] = chla
                pos = cuda.atomic.add(sh_n, 0, 1)              # lugar dentro del bloque
            if not math.isnan(ndvi):
                sh[5, tid] = ndvi
                sh[6, tid] = 1.0
            if not math.isnan(ndwi):
                sh[7, tid] = ndwi
                sh[8, tid] = 1.0
            if not math.isnan(fai):
                sh[10, tid] = 1.0
                if fai > UMBRAL_FAI:
                    sh[11, tid] = 1.0
    cuda.syncthreads()

    if tid == 0:
        sh_base[0] = cuda.atomic.add(contador, f, sh_n[0])     # tramo del bloque

    # Reducción en árbol: 256 -> 128 -> ... -> 1
    paso = HILOS // 2
    while paso > 0:
        if tid < paso:
            for q in range(Q):
                sh[q, tid] += sh[q, tid + paso]
            if sh_max[tid + paso] > sh_max[tid]:
                sh_max[tid] = sh_max[tid + paso]
        cuda.syncthreads()
        paso //= 2

    if pos >= 0:
        valores[f, sh_base[0] + pos] = chla
    if tid == 0:
        for q in range(Q):
            cuda.atomic.add(acum, (f, q), sh[q, 0])
        cuda.atomic.max(maximo, f, sh_max[0])


def mascara_lago_gpu(d_mascaras):
    """Máscara fija del lago (uint8, en la GPU) a partir de la capa `agua` de K1."""
    _, _, alto, ancho = d_mascaras.shape
    d_lago = cuda.device_array((alto, ancho), dtype=np.uint8)
    grid, bloque = configuracion(1, alto, ancho)
    k_mascara_lago[grid[:2], bloque](d_mascaras, d_lago)
    return d_lago


def estadisticas_gpu(d_capas, d_mascaras) -> dict:
    """Máscara del lago y K2 sobre las salidas de K1 que están en la GPU.

    Returns:
        dict con, en el host: `lago` (uint8 alto x ancho), `acum` (fechas x Q, float64),
        `maximo` (fechas,), `valores` (lista por fecha con la clorofila-a usable) y
        `tiempos` (eventos de CUDA, s): mascara_s, k2_s, copia_gpu_cpu_s.
    """
    n_fechas, _, alto, ancho = d_capas.shape
    ev = [cuda.event() for _ in range(4)]

    ev[0].record()
    d_lago = mascara_lago_gpu(d_mascaras)
    ev[1].record()

    d_acum = cuda.to_device(np.zeros((n_fechas, Q), dtype=np.float64))
    d_max = cuda.to_device(np.full(n_fechas, MENOS_INF, dtype=np.float32))
    d_valores = cuda.device_array((n_fechas, alto * ancho), dtype=np.float32)
    d_contador = cuda.to_device(np.zeros(n_fechas, dtype=np.int32))
    grid, bloque = configuracion(n_fechas, alto, ancho)
    k2_reducciones[grid, bloque](d_capas, d_mascaras, d_lago, d_acum, d_max, d_valores, d_contador)
    ev[2].record()

    acum = d_acum.copy_to_host()
    maximo = d_max.copy_to_host()
    contador = d_contador.copy_to_host()
    lago = d_lago.copy_to_host()
    valores = [
        d_valores[f, : contador[f]].copy_to_host() if contador[f] else np.empty(0, np.float32)
        for f in range(n_fechas)
    ]
    ev[3].record()
    ev[3].synchronize()

    return {
        "lago": lago, "d_lago": d_lago, "acum": acum, "maximo": maximo, "valores": valores,
        "tiempos": {
            "mascara_s": ev[0].elapsed_time(ev[1]) / 1000,
            "k2_s": ev[1].elapsed_time(ev[2]) / 1000,
            "copia_gpu_cpu_s": ev[2].elapsed_time(ev[3]) / 1000,
        },
    }


# --------------------------------------------------------------------------
# K3: histogramas de clorofila-a por fecha
# --------------------------------------------------------------------------
@cuda.jit
def k3_histograma(capas, mascaras, lago, bordes, hist):
    """Histograma privado por bloque en memoria compartida, luego combinado.

    Cada hilo suma su píxel al histograma de su bloque (atómico en memoria compartida,
    rápido); al final cada bloque suma sus bins no vacíos a `hist[fecha]` con un atómico
    global por bin. El bin se elige como np.histogram: estimación por división y
    corrección contra los mismos bordes float32, así el resultado es idéntico.
    """
    col, fila = cuda.grid(2)
    f = cuda.blockIdx.z
    tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    sh = cuda.shared.array(shape=N_BINS, dtype=int32)
    if tid < N_BINS:
        sh[tid] = 0
    cuda.syncthreads()

    if fila < lago.shape[0] and col < lago.shape[1]:
        if (lago[fila, col] != 0) and (mascaras[f, VALIDOS, fila, col] != 0):
            x = capas[f, CHLA, fila, col]
            if not math.isnan(x):
                inicio = bordes[0]
                fin = bordes[N_BINS]
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
                cuda.atomic.add(sh, b, 1)
    cuda.syncthreads()

    if tid < N_BINS and sh[tid] > 0:
        cuda.atomic.add(hist, (f, tid), sh[tid])


def histogramas_gpu(d_capas, d_mascaras, d_lago, bordes: np.ndarray) -> tuple[np.ndarray, dict]:
    """K3 sobre las capas de K1 y la máscara del lago que ya están en la GPU.

    Returns:
        (hist int64 (fechas, N_BINS) en el host, tiempos con eventos de CUDA).
    """
    n_fechas, _, alto, ancho = d_capas.shape
    ev = [cuda.event() for _ in range(3)]
    ev[0].record()
    d_bordes = cuda.to_device(bordes.astype(np.float32))
    d_hist = cuda.to_device(np.zeros((n_fechas, N_BINS), dtype=np.int32))
    grid, bloque = configuracion(n_fechas, alto, ancho)
    k3_histograma[grid, bloque](d_capas, d_mascaras, d_lago, d_bordes, d_hist)
    ev[1].record()
    hist = d_hist.copy_to_host().astype(np.int64)
    ev[2].record()
    ev[2].synchronize()
    return hist, {"k3_s": ev[0].elapsed_time(ev[1]) / 1000,
                  "copia_gpu_cpu_s": ev[1].elapsed_time(ev[2]) / 1000}

