"""Pasos 1 y 2 del enunciado para la Parte I: operaciones y tiempo secuencial.

Mide, con el protocolo de `medicion/tiempos.py`, cada operación del pipeline
secuencial del laboratorio (`lab4/indices.py` y `lab4/procesar.py`, sin cambios)
sobre todas las fechas de cada lago, y genera la tabla de operaciones repetitivas
con el tamaño real de los rásteres.

Fases medidas por lago (cada una recorre las 11 fechas):

    Lectura (lab4 leer_bandas)      GeoTIFF int16 -> float32, nodata -> NaN, / 10 000
    Mascara de validos (SCL)        indices.mascara_valida
    NDVI, NDWI, NDCI, FAI           indices.ndvi / ndwi / ndci / fai
    Clorofila-a                     indices.clorofila_a (polinomio de grado 3)
    Mascara de agua (wbi)           indices.mascara_agua (7 índices + 8 umbrales)
    K1 completo                     todo lo anterior salvo la lectura (= lo que va a GPU)
    Mascara fija del lago           agua en >= 60 % de las fechas
    K2 estadisticas por fecha       procesar.resumen (medias, mediana, p90, % alto)

Todo se calcula en memoria (no se reescriben los GeoTIFF de data/processed).

    .venv\\Scripts\\python -m src.referencia_cpu.parte1_secuencial
    .venv\\Scripts\\python -m src.referencia_cpu.parte1_secuencial --repeticiones 3

Salidas:
    resultados/tiempos/parte1_secuencial.csv
    resultados/analisis/parte1_operaciones.csv    (tabla de operaciones repetitivas)
    resultados/analisis/parte1_tamanos.csv        (malla y píxeles por lago)

Responsable: Javier Linares.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import src  # noqa: F401  (agrega lab4/ al sys.path)
import indices as ix
import procesar
from config import FECHAS
from src import RESULTADOS
from src.medicion.tiempos import guardar, medir

VERSION = "CPU secuencial"

# --------------------------------------------------------------------------
# Tabla de operaciones (paso 1). Conteo por píxel y por fecha, leído del código de
# lab4/indices.py. "aritmeticas" = + - * / y potencias; "logicas" = comparaciones,
# &, |, ~, isnan; "selecciones" = np.where / división condicionada (where=).
# --------------------------------------------------------------------------
OPERACIONES = [
    {
        "operacion": "Lectura y reflectancia", "funcion_lab4": "indices.leer_bandas",
        "entradas": "9 bandas L1C int16 + SCL", "salidas": "10 capas float32",
        "aritmeticas": 9, "logicas": 10, "selecciones": 10,
        "detalle": "astype float32, nodata -> NaN (1 comparación + 1 where por capa), / 10 000 en las 9 bandas",
    },
    {
        "operacion": "Mascara de validos", "funcion_lab4": "indices.mascara_valida",
        "entradas": "B04, SCL", "salidas": "bool",
        "aritmeticas": 0, "logicas": 18, "selecciones": 0,
        "detalle": "2 isnan + 2 NOT + AND; isin con 6 clases SCL (6 comparaciones + 5 OR) + NOT; AND final",
    },
    {
        "operacion": "NDVI", "funcion_lab4": "indices.ndvi",
        "entradas": "B08, B04", "salidas": "float32",
        "aritmeticas": 3, "logicas": 4, "selecciones": 2,
        "detalle": "(a-b)/(a+b) con a>0, b>0 y denominador != 0",
    },
    {
        "operacion": "NDWI", "funcion_lab4": "indices.ndwi",
        "entradas": "B03, B08", "salidas": "float32",
        "aritmeticas": 3, "logicas": 4, "selecciones": 2,
        "detalle": "(a-b)/(a+b) con a>0, b>0 y denominador != 0",
    },
    {
        "operacion": "NDCI", "funcion_lab4": "indices.ndci",
        "entradas": "B05, B04", "salidas": "float32",
        "aritmeticas": 3, "logicas": 4, "selecciones": 2,
        "detalle": "(a-b)/(a+b) con a>0, b>0 y denominador != 0",
    },
    {
        "operacion": "FAI", "funcion_lab4": "indices.fai",
        "entradas": "B04, B07, B8A", "salidas": "float32",
        "aritmeticas": 5, "logicas": 0, "selecciones": 0,
        "detalle": "B07 - B04 - (B8A - B04) * 118 / 200",
    },
    {
        "operacion": "Clorofila-a", "funcion_lab4": "indices.clorofila_a",
        "entradas": "NDCI", "salidas": "float32",
        "aritmeticas": 8, "logicas": 1, "selecciones": 0,
        "detalle": "826.57 n^3 - 176.43 n^2 + 19 n + 4.071, recorte en 0",
    },
    {
        "operacion": "Mascara de agua (wbi)", "funcion_lab4": "indices.mascara_agua",
        "entradas": "B02, B03, B04, B08, B11, B12", "salidas": "bool",
        "aritmeticas": 29, "logicas": 22, "selecciones": 6,
        "detalle": "5 diferencias normalizadas, AWEIsh, AWEInsh, DBSI; 6 umbrales con OR, filtro UABS (2 umbrales), isnan",
    },
    {
        "operacion": "Agua & validos", "funcion_lab4": "indices.calcular_indices",
        "entradas": "wbi, validos", "salidas": "bool",
        "aritmeticas": 0, "logicas": 1, "selecciones": 0,
        "detalle": "AND de las dos máscaras",
    },
    {
        "operacion": "Mascara fija del lago", "funcion_lab4": "procesar.construir_mascara_lago",
        "entradas": "agua de las 11 fechas", "salidas": "bool (1 por lago)",
        "aritmeticas": 1, "logicas": 0, "selecciones": 0,
        "detalle": "suma acumulada por fecha (1 suma por píxel y fecha); al final / 11 y >= 0.6",
    },
    {
        "operacion": "K2 estadisticas por fecha", "funcion_lab4": "procesar.resumen",
        "entradas": "capas del espejo de agua", "salidas": "15 valores por fecha",
        "aritmeticas": 6, "logicas": 5, "selecciones": 0,
        "detalle": "4 medias (nanmean), mediana y p90 (ordenamiento parcial), máximo, 2 porcentajes sobre umbral; sobre lago & validos",
    },
]


# --------------------------------------------------------------------------
# Fases (cada función recorre todas las fechas de un lago)
# --------------------------------------------------------------------------
def leer_todas(lago: str) -> list[dict]:
    return [
        ix.leer_bandas(procesar.ruta_cruda(lago, f, "L1C"), procesar.ruta_cruda(lago, f, "L2A"))
        for f in FECHAS[lago]
    ]


def k1_completo(bandas: dict) -> dict:
    """`indices.calcular_indices` sin la lectura (exactamente las mismas funciones)."""
    validos = ix.mascara_valida(bandas)
    capas = {"ndvi": ix.ndvi(bandas), "ndwi": ix.ndwi(bandas),
             "ndci": ix.ndci(bandas), "fai": ix.fai(bandas)}
    capas["chla"] = ix.clorofila_a(capas["ndci"])
    capas["agua"] = ix.mascara_agua(bandas) & validos
    capas["validos"] = validos
    return capas


def mascara_lago(capas_por_fecha: list[dict]) -> np.ndarray:
    """Igual que `procesar.construir_mascara_lago`, pero en memoria."""
    acumulado = None
    for capas in capas_por_fecha:
        agua = capas["agua"].astype(np.float32)
        acumulado = agua if acumulado is None else acumulado + agua
    return (acumulado / len(capas_por_fecha)) >= procesar.FRACCION_AGUA_LAGO


def medir_lago(lago: str, repeticiones: int) -> tuple[list, dict]:
    fechas = FECHAS[lago]
    bandas = leer_todas(lago)
    capas = [k1_completo(b) for b in bandas]
    mascara = mascara_lago(capas)

    # La máscara en memoria debe ser la misma que la guardada por el laboratorio.
    ruta = procesar.ruta_mascara(lago)
    if ruta.exists():
        distintos = int((procesar.leer_mascara_lago(lago) != mascara).sum())
        print(f"  [validación] máscara del lago en memoria vs {ruta.name}: {distintos} píxeles distintos")

    alto, ancho = bandas[0]["B04"].shape
    tam = {
        "lago": lago, "alto": alto, "ancho": ancho, "pixeles_por_fecha": alto * ancho,
        "fechas": len(fechas), "pixeles_fecha_total": alto * ancho * len(fechas),
        "pixeles_espejo_agua": int(mascara.sum()),
        "pixeles_usables_total": int(sum((mascara & c["validos"]).sum() for c in capas)),
        "megabytes_bandas_float32": round(alto * ancho * 10 * 4 * len(fechas) / 1e6, 1),
    }
    nota = f"{lago}; {len(fechas)} fechas; malla {alto}x{ancho}"

    def fase(nombre: str, funcion) -> object:
        r = medir(funcion, fase=nombre, version=VERSION, repeticiones=repeticiones,
                  notas=f"{nota}; lab4 sin cambios")
        print(f"  {nombre:32s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
        return r

    mediciones = [
        fase(f"Lectura {lago}", lambda: leer_todas(lago)),
        fase(f"Mascara de validos {lago}", lambda: [ix.mascara_valida(b) for b in bandas]),
        fase(f"NDVI {lago}", lambda: [ix.ndvi(b) for b in bandas]),
        fase(f"NDWI {lago}", lambda: [ix.ndwi(b) for b in bandas]),
        fase(f"NDCI {lago}", lambda: [ix.ndci(b) for b in bandas]),
        fase(f"FAI {lago}", lambda: [ix.fai(b) for b in bandas]),
        fase(f"Clorofila-a {lago}", lambda: [ix.clorofila_a(c["ndci"]) for c in capas]),
        fase(f"Mascara de agua {lago}", lambda: [ix.mascara_agua(b) for b in bandas]),
        fase(f"K1 completo {lago}", lambda: [k1_completo(b) for b in bandas]),
        fase(f"Mascara fija del lago {lago}", lambda: mascara_lago(capas)),
        fase(f"K2 estadisticas por fecha {lago}", lambda: [
            procesar.resumen(lago, f, c, mascara) for f, c in zip(fechas, capas)
        ]),
    ]
    return mediciones, tam


def tabla_operaciones(tamanos: pd.DataFrame) -> pd.DataFrame:
    """Operaciones repetitivas: conteo por píxel × píxeles × fechas, por lago."""
    filas = []
    for op in OPERACIONES:
        for _, t in tamanos.iterrows():
            # K2 solo recorre el espejo de agua; la máscara fija se calcula una vez.
            if op["operacion"].startswith("K2"):
                datos, repeticiones = "píxeles usables del lago", int(t["pixeles_usables_total"])
            else:
                datos, repeticiones = "píxeles de la malla × fechas", int(t["pixeles_fecha_total"])
            por_pixel = op["aritmeticas"] + op["logicas"] + op["selecciones"]
            filas.append({
                **op, "lago": t["lago"], "datos_procesados": datos,
                "repeticiones": repeticiones,
                "operaciones_por_pixel": por_pixel,
                "operaciones_totales": por_pixel * repeticiones,
            })
    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perfil secuencial de la Parte I")
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()

    mediciones, tamanos = [], []
    for lago in FECHAS:
        print(f"\n=== {lago} ===")
        m, t = medir_lago(lago, args.repeticiones)
        mediciones += m
        tamanos.append(t)

    guardar(mediciones, "parte1_secuencial.csv", sobrescribir=True)
    analisis = RESULTADOS / "analisis"
    analisis.mkdir(parents=True, exist_ok=True)
    tamanos = pd.DataFrame(tamanos)
    tamanos.to_csv(analisis / "parte1_tamanos.csv", index=False)
    tabla_operaciones(tamanos).to_csv(analisis / "parte1_operaciones.csv", index=False)
    print("\n" + tamanos.to_string(index=False))
    print("\nGuardado: resultados/tiempos/parte1_secuencial.csv, "
          "resultados/analisis/parte1_{tamanos,operaciones}.csv")


if __name__ == "__main__":
    main()
