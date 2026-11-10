"""Regresión Logística propia en GPU (Numba CUDA + CuPy). Responsable: María José Yee.

Descenso de gradiente por mini-lotes sobre X float32 estandarizada,
pérdida logística ponderada (equivalente a class_weight="balanced") con
penalización L2 (lambda = 1 / (C * n)). Kernels propios: sigmoide + gradiente
por bloque con reducción en memoria compartida.
"""

# TODO(María José): RegresionLogisticaGPU(C, lr, epocas, tam_lote).fit / predict_proba
