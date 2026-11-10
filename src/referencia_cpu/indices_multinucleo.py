"""Índices por píxel con Numba @njit(parallel=True) y prange.

Misma lógica y mismo contrato que `calculo_gpu.kernels_indices`, para comparar
CPU secuencial (lab4/indices.py) vs CPU multinúcleo vs GPU. Responsable: Angie Vela.
"""

# TODO(Angie): indices_prange(cubo_fecha, scl_fecha) -> dict de capas
