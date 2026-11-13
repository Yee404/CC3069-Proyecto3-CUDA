"""Fórmulas de K1 para UN píxel, compartidas por la GPU y el CPU multinúcleo.

Es la misma matemática de `lab4/indices.py`, pero escrita por píxel (escalares)
en lugar de por arreglo. Este archivo no importa CUDA ni lee archivos: es Python
puro que se compila dos veces,

    calculo_gpu/kernels_indices.py        -> cuda.jit(device=True)(pixel)
    referencia_cpu/indices_multinucleo.py -> numba.njit(pixel)

así las versiones GPU y CPU multinúcleo ejecutan exactamente las mismas
operaciones y cualquier diferencia de tiempo es del hardware, no del algoritmo.

Reglas para replicar NumPy (float32):

* Todas las constantes son float32. En NumPy 2 un escalar de Python que opera con
  un arreglo float32 se convierte a float32 (NEP 50), así que el laboratorio
  calcula todo en float32; aquí se fuerza lo mismo para que Numba no promueva a
  float64.
* Las comparaciones con NaN son falsas, igual que en NumPy.
* `np.clip(chla, 0, None)` conserva NaN; por eso el recorte se escribe como
  `0 si chla < 0`, no como `max(chla, 0)`.
* Sin ramas de control: las condiciones se combinan con `|` y `&` sobre booleanos
  y las selecciones con expresiones condicionales, para reducir divergencia.

Salida de `pixel(...)`, en este orden (ver CAPAS_FLOAT y CAPAS_MASCARA):

    ndvi, ndwi, ndci, fai, chla   float32 (NaN donde el laboratorio da NaN)
    agua, validos, y_alta         bool

`agua` = máscara de agua del script CyanoLakes & válidos (igual que la capa
"agua" de `indices.calcular_indices`); `y_alta` = chla > 10 mg/m3 (variable
respuesta de `dataset_ml`).

Responsable: Nadissa.
"""

from __future__ import annotations

import math

import numpy as np

import src  # noqa: F401  (agrega lab4/ al sys.path)
import dataset_ml as _dml
import indices as _ix

F32 = np.float32

# Orden de las capas en los arreglos de salida.
CAPAS_FLOAT = ("ndvi", "ndwi", "ndci", "fai", "chla")
CAPAS_MASCARA = ("agua", "validos", "y_alta")

# Clases SCL inválidas del laboratorio (0, 1, 3, 8, 9, 10). `leer_cubo` guarda el
# "sin dato" de SCL como 0, que ya es inválido, así que no hace falta otro chequeo.
assert tuple(_ix.SCL_INVALIDAS) == (0, 1, 3, 8, 9, 10)

NAN = F32(np.nan)
CERO = F32(0.0)

# Umbrales de la máscara de agua (lab4/indices.py)
MNDWI_UMBRAL = F32(_ix.MNDWI_THRESHOLD)   # 0.42
NDWI_UMBRAL = F32(_ix.NDWI_THRESHOLD)     # 0.4
AWEINSH_UMBRAL = F32(0.1879)
AWEISH_UMBRAL = F32(0.1112)
VEGETACION_UMBRAL = F32(-0.2)
HOJAS_UMBRAL = F32(1.0)
UABS_AWEINSH = F32(-0.03)

# Coeficientes
K_FAI = F32((783 - 665) / (865 - 665))    # 0.59
C3, C2, C1, C0 = F32(826.57), F32(176.43), F32(19.0), F32(4.071)
DOS_Y_MEDIO, UNO_Y_MEDIO, UN_CUARTO = F32(2.5), F32(1.5), F32(0.25)
CUATRO, DOS_TRES_CUARTOS = F32(4.0), F32(2.75)

UMBRAL_CHLA = F32(_dml.UMBRAL_CHLA_OMS)  # 10 mg/m3


# Los auxiliares del laboratorio (`_div`, `_normalizado`) se escriben en línea:
# Numba solo puede llamar funciones compiladas, y así el mismo código sirve para
# `njit` y para `cuda.jit(device=True)` sin compilar funciones auxiliares aparte.
#
#   _div(n, d)         -> n / d si d != 0, si no NaN
#   _normalizado(a, b) -> _div(a - b, a + b) si a > 0 y b > 0, si no NaN


def pixel(b02, b03, b04, b05, b07, b08, b8a, b11, b12, scl):
    # Máscara de válidos (indices.mascara_valida)
    validos = (not math.isnan(b04)) & (scl != 0) & (scl != 1) & (scl != 3) \
        & (scl != 8) & (scl != 9) & (scl != 10)

    # Índices normalizados (indices._normalizado)
    d = b08 + b04
    ndvi = ((b08 - b04) / d if d != CERO else NAN) if (b08 > CERO) & (b04 > CERO) else NAN
    d = b03 + b08
    ndwi = ((b03 - b08) / d if d != CERO else NAN) if (b03 > CERO) & (b08 > CERO) else NAN
    d = b05 + b04
    ndci = ((b05 - b04) / d if d != CERO else NAN) if (b05 > CERO) & (b04 > CERO) else NAN

    # FAI (indices.fai)
    fai = b07 - b04 - (b8a - b04) * K_FAI

    # Clorofila-a (polinomio del script, recortado en 0 conservando NaN)
    chla = C3 * ndci ** 3 - C2 * ndci ** 2 + C1 * ndci + C0
    chla = CERO if chla < CERO else chla

    # Máscara de agua `wbi` (indices.mascara_agua, con indices._div):
    # azul=B02, verde=B03, rojo=B04, nir=B08, swir1=B11, swir2=B12
    d = b08 + b04
    vegetacion = (b08 - b04) / d if d != CERO else NAN
    d = b03 + b11
    mndwi = (b03 - b11) / d if d != CERO else NAN
    d = b03 + b08
    indice_agua = (b03 - b08) / d if d != CERO else NAN
    d = b08 + b11
    ndwi_hojas = (b08 - b11) / d if d != CERO else NAN
    aweish = b02 + DOS_Y_MEDIO * b03 - UNO_Y_MEDIO * (b08 + b11) - UN_CUARTO * b12
    aweinsh = CUATRO * (b03 - b11) - (UN_CUARTO * b08 + DOS_TRES_CUARTOS * b11)
    d = b11 + b03
    dbsi = ((b11 - b03) / d if d != CERO else NAN) - vegetacion

    es_agua = (mndwi > MNDWI_UMBRAL) | (indice_agua > NDWI_UMBRAL) \
        | (aweinsh > AWEINSH_UMBRAL) | (aweish > AWEISH_UMBRAL) \
        | (vegetacion < VEGETACION_UMBRAL) | (ndwi_hojas > HOJAS_UMBRAL)
    es_agua = es_agua & (not ((aweinsh <= UABS_AWEINSH) | (dbsi > CERO)))
    es_agua = es_agua & (not math.isnan(b04))

    agua = es_agua & validos
    y_alta = chla > UMBRAL_CHLA
    return ndvi, ndwi, ndci, fai, chla, agua, validos, y_alta
