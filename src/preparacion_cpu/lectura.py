"""Etapa 1: lectura de rasters en paralelo por fecha (CPU multinúcleo).

Contrato de salida (lo que consume `calculo_gpu.kernels_indices`):

    cubo:   np.ndarray float32, forma (n_fechas, n_bandas, alto, ancho), C-contiguo
            orden de bandas = config.BANDAS_L1C (B02 B03 B04 B05 B07 B08 B8A B11 B12)
            reflectancia 0-1 (ya dividida entre ESCALA_REFLECTANCIA), NaN = sin dato
    scl:    np.ndarray uint8, forma (n_fechas, alto, ancho)
    fechas: list[str]

Un lago por llamada (Atitlán y Amatitlán tienen mallas distintas).
Versiones a medir: secuencial (bucle por fecha) y multinúcleo
(concurrent.futures.ProcessPoolExecutor, un proceso por fecha).
"""

# TODO(Javier): leer_cubo(lago, n_procesos=1) -> (cubo, scl, fechas, perfil)
