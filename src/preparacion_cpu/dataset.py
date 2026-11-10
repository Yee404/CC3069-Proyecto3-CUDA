"""Etapa 4: dataset tabular, bloques de 1 km y particiones (CPU).

Reutiliza `dataset_ml` del Laboratorio 4. Las particiones se generan UNA vez
y se guardan (índices) para que CPU y GPU entrenen exactamente con las mismas filas:

    - train/test 70/30 estratificado (semilla 42)
    - StratifiedKFold(3) interno para la búsqueda de hiperparámetros
    - validación aleatoria (5 pliegues), espacial por bloque_1km (5) y temporal por escena (5)
    - corte cronológico (8 fechas antiguas / 3 recientes por lago)
    - experimentos entre lagos (Atitlán -> Amatitlán y viceversa) e intra-lago

Responsable: Javier Linares.
"""

# TODO(Javier): construir_particiones(datos) -> dict[str, list[(idx_ent, idx_val)]]
# TODO(Javier): a_matrices_float32(datos) -> (X float32 C-contiguo, y int8)
