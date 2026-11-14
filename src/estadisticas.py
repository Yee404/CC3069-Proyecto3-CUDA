"""Armado de la tabla de estadísticas por fecha a partir de sumas y conteos.

Python puro (sin CUDA ni archivos), compartido por la versión GPU y la CPU
multinúcleo de K2. Produce exactamente las columnas de `procesar.resumen` del
laboratorio, para compararlas una a una.

Cada versión de K2 entrega, por fecha, las cantidades de CANTIDADES (sumas y
conteos sobre los píxeles usables = máscara fija del lago & válidos), el máximo de
clorofila-a y la mediana y el percentil 90 de clorofila-a.

Responsable: Nadissa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import src  # noqa: F401  (agrega lab4/ al sys.path)
import procesar
from config import NUBOSIDAD

# Orden de las cantidades acumuladas por K2 (columnas del arreglo `acum`).
CANTIDADES = (
    "n_usable",
    "suma_ndci", "n_ndci",
    "suma_chla", "n_chla",
    "suma_ndvi", "n_ndvi",
    "suma_ndwi", "n_ndwi",
    "n_chla_alto",       # chla >= procesar.UMBRAL_CHLA_ALTO (20 mg/m3)
    "n_fai", "n_fai_alto",  # fai > procesar.UMBRAL_FAI (0.08)
)
Q = len(CANTIDADES)
I = {nombre: i for i, nombre in enumerate(CANTIDADES)}

UMBRAL_CHLA_ALTO = np.float32(procesar.UMBRAL_CHLA_ALTO)
UMBRAL_FAI = np.float32(procesar.UMBRAL_FAI)
FRACCION_AGUA_LAGO = np.float32(procesar.FRACCION_AGUA_LAGO)

# Columnas de procesar.resumen, en el mismo orden.
COLUMNAS = [
    "lago", "fecha", "nubosidad_pct", "area_lago_km2", "cobertura_valida_pct",
    "pixeles_analizados", "ndci_medio", "chla_media", "chla_mediana", "chla_p90",
    "chla_max", "ndvi_medio", "ndwi_medio", "pct_alto", "pct_fai_alto",
]


def _cociente(a: float, b: float, factor: float = 1.0) -> float:
    return a / b * factor if b else np.nan


def armar_tabla(
    lago: str,
    fechas: list[str],
    n_lago: int,
    acum: np.ndarray,
    maximo: np.ndarray,
    medianas: np.ndarray,
    p90: np.ndarray,
) -> pd.DataFrame:
    """Una fila por fecha con las columnas de `procesar.resumen`."""
    filas = []
    for f, fecha in enumerate(fechas):
        a = acum[f]
        n_chla = a[I["n_chla"]]
        filas.append({
            "lago": lago,
            "fecha": pd.Timestamp(fecha),
            "nubosidad_pct": NUBOSIDAD[lago][fecha],
            "area_lago_km2": float(n_lago * procesar.AREA_PIXEL_KM2),
            "cobertura_valida_pct": _cociente(a[I["n_usable"]], n_lago, 100),
            "pixeles_analizados": int(n_chla),
            "ndci_medio": _cociente(a[I["suma_ndci"]], a[I["n_ndci"]]),
            "chla_media": _cociente(a[I["suma_chla"]], n_chla),
            "chla_mediana": float(medianas[f]) if n_chla else np.nan,
            "chla_p90": float(p90[f]) if n_chla else np.nan,
            "chla_max": float(maximo[f]) if n_chla else np.nan,
            "ndvi_medio": _cociente(a[I["suma_ndvi"]], a[I["n_ndvi"]]),
            "ndwi_medio": _cociente(a[I["suma_ndwi"]], a[I["n_ndwi"]]),
            "pct_alto": _cociente(a[I["n_chla_alto"]], n_chla, 100),
            "pct_fai_alto": _cociente(a[I["n_fai_alto"]], a[I["n_fai"]], 100),
        })
    return pd.DataFrame(filas, columns=COLUMNAS)


def orden_en_cpu(valores_por_fecha: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mediana y percentil 90 de cada fecha con NumPy (mismo método que el laboratorio)."""
    medianas = np.array([np.median(v) if v.size else np.nan for v in valores_por_fecha])
    p90 = np.array([np.percentile(v, 90) if v.size else np.nan for v in valores_por_fecha])
    return medianas, p90
