"""K2 (reducciones) y K3 (histogramas) con Numba CUDA. Responsable: Angie Vela.

K2: chla media, % del lago sobre el umbral y píxeles válidos por lago y fecha
    (reducción en memoria compartida por bloque + cuda.atomic.add final).
K3: histograma privado por bloque en memoria compartida, luego combinado.
Validar contra lab4/procesar.resumen().
"""

# TODO(Angie): k2_reduccion, k3_histograma
