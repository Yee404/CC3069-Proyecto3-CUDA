"""K1: índices por píxel con Numba CUDA. Responsable: Angie Vela.

Un hilo por píxel, bloques 2D de 16x16, grid que cubre (alto, ancho) del recorte.
Salidas: NDVI, NDWI, NDCI, FAI, clorofila-a, máscara de agua (wbi & válidos) y
y_alta = chla > 10 mg/m3. Debe coincidir con lab4/indices.py (ver PLAN.md §8).
"""

# TODO(Angie): @cuda.jit k1_indices(bandas, scl, salidas)
# TODO(Angie): calcular_indices_gpu(cubo, scl, usar_streams=True) -> capas + tiempos por fase
