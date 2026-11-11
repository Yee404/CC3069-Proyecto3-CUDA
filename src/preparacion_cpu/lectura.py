"""Etapa 1: lectura de rasters en paralelo por fecha (CPU multinúcleo).

Contrato de salida (lo que consume `calculo_gpu.kernels_indices`):

    cubo:   np.ndarray float32, forma (n_fechas, n_bandas, alto, ancho), C-contiguo
            orden de bandas = config.BANDAS_L1C (B02 B03 B04 B05 B07 B08 B8A B11 B12)
            reflectancia 0-1 (ya dividida entre ESCALA_REFLECTANCIA), NaN = sin dato
    scl:    np.ndarray uint8, forma (n_fechas, alto, ancho)
            clase SCL de L2A; el "sin dato" del archivo se guarda como 0 (NO_DATA),
            que ya está en `indices.SCL_INVALIDAS`, así que la máscara de válidos
            no cambia respecto al laboratorio
    fechas: list[str]
    perfil: dict con crs, transform, bounds, height y width (igual que
            `indices.leer_perfil`), para reinsertar resultados en la malla

Un lago por llamada (Atitlán y Amatitlán tienen mallas distintas).

Los valores son bit a bit iguales a los de `lab4/indices.leer_bandas`: misma
conversión int16 -> float32, mismo reemplazo de nodata por NaN y misma división
en float32 entre 10 000 (ver `validar_contra_lab4`).

Versiones a medir (parámetro `modo`):

    "secuencial" : bucle por fecha, un núcleo (equivale a cómo lee el laboratorio)
    "procesos"   : ProcessPoolExecutor, un proceso por fecha. Cada proceso escribe su
                   fecha directamente en un bloque de memoria compartida
                   (multiprocessing.shared_memory), así que no se serializan los
                   arreglos de vuelta; al final se hace una sola copia al arreglo
                   de salida
    "hilos"      : ThreadPoolExecutor, un hilo por fecha. GDAL (rasterio) suelta el
                   GIL al descomprimir, así que los hilos también corren en paralelo
                   y evitan el costo de arrancar procesos en Windows

Uso desde la raíz del repositorio:

    .venv\\Scripts\\python -m src.preparacion_cpu.lectura             # valida y mide
    .venv\\Scripts\\python -m src.preparacion_cpu.lectura --evidencia 40
        # repite la lectura multinúcleo ~40 s para capturar el Administrador de Tareas

Responsable: Javier Linares.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import rasterio

import src  # noqa: F401  (agrega lab4/ al sys.path)
from config import BANDAS_L1C, ESCALA_REFLECTANCIA, FECHAS
from procesar import ruta_cruda

MODOS = ("secuencial", "procesos", "hilos")
N_BANDAS = len(BANDAS_L1C)

# Valor con el que se guarda el "sin dato" de SCL en el arreglo uint8.
SCL_SIN_DATO = 0


# --------------------------------------------------------------------------
# Lectura de una fecha
# --------------------------------------------------------------------------
def forma_malla(lago: str) -> tuple[int, int]:
    """(alto, ancho) de la malla del lago, tomada de su primera fecha."""
    with rasterio.open(ruta_cruda(lago, FECHAS[lago][0], "L1C")) as src:
        return src.height, src.width


def leer_perfil(lago: str) -> dict:
    """Perfil de la malla del lago (mismo formato que `indices.leer_perfil`)."""
    with rasterio.open(ruta_cruda(lago, FECHAS[lago][0], "L1C")) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "height": src.height,
            "width": src.width,
        }


def _leer_l1c(ruta: Path, destino: np.ndarray) -> None:
    """Escribe en `destino` (n_bandas, alto, ancho) la reflectancia L1C.

    Replica `indices._leer` + `indices.leer_bandas`: int16 -> float32, nodata -> NaN
    y división en float32 entre ESCALA_REFLECTANCIA. Las bandas se ordenan según
    BANDAS_L1C usando la descripción de cada banda del GeoTIFF.
    """
    with rasterio.open(ruta) as src:
        datos = src.read().astype(np.float32)
        nodata = src.nodata
        nombres = list(src.descriptions)

    if destino.shape[1:] != datos.shape[1:]:
        raise ValueError(
            f"{ruta.name}: malla {datos.shape[1:]} distinta de la del lago {destino.shape[1:]}"
        )
    for i, banda in enumerate(BANDAS_L1C):
        capa = datos[nombres.index(banda)]
        if nodata is not None:
            capa = np.where(capa == nodata, np.nan, capa)
        np.divide(capa, ESCALA_REFLECTANCIA, out=destino[i])


def _leer_scl(ruta: Path, destino: np.ndarray) -> None:
    """Escribe en `destino` (alto, ancho) la banda SCL como uint8."""
    with rasterio.open(ruta) as src:
        nombres = list(src.descriptions)
        indice = nombres.index("SCL") + 1 if "SCL" in nombres else 1
        datos = src.read(indice)
        nodata = src.nodata

    if nodata is not None:
        datos = np.where(datos == nodata, SCL_SIN_DATO, datos)
    if np.issubdtype(datos.dtype, np.floating):
        datos = np.where(np.isfinite(datos), datos, SCL_SIN_DATO)
    destino[...] = datos.astype(np.uint8)


def _leer_fecha(lago: str, fecha: str, cubo_fecha: np.ndarray, scl_fecha: np.ndarray) -> None:
    _leer_l1c(ruta_cruda(lago, fecha, "L1C"), cubo_fecha)
    _leer_scl(ruta_cruda(lago, fecha, "L2A"), scl_fecha)


def _leer_fecha_compartida(args: tuple) -> str:
    """Tarea de un proceso: lee una fecha y la escribe en la memoria compartida."""
    lago, fecha, i, nombre_cubo, forma_cubo, nombre_scl, forma_scl = args
    shm_cubo = shared_memory.SharedMemory(name=nombre_cubo)
    shm_scl = shared_memory.SharedMemory(name=nombre_scl)
    try:
        cubo = np.ndarray(forma_cubo, dtype=np.float32, buffer=shm_cubo.buf)
        scl = np.ndarray(forma_scl, dtype=np.uint8, buffer=shm_scl.buf)
        _leer_fecha(lago, fecha, cubo[i], scl[i])
        del cubo, scl  # soltar las vistas antes de cerrar la memoria compartida
    finally:
        shm_cubo.close()
        shm_scl.close()
    return fecha


# --------------------------------------------------------------------------
# Lectura del cubo completo
# --------------------------------------------------------------------------
def _n_trabajadores(n_procesos: int | None, n_fechas: int) -> int:
    n = n_procesos or os.cpu_count() or 1
    return max(1, min(n, n_fechas))


def leer_cubo(
    lago: str,
    n_procesos: int | None = None,
    modo: str = "procesos",
    fechas: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Lee todas las fechas de un lago como un cubo float32 contiguo.

    Args:
        lago: "Atitlan" o "Amatitlan".
        n_procesos: trabajadores para "procesos"/"hilos" (None = todos los núcleos
            lógicos, acotado al número de fechas). Con `modo="secuencial"` se ignora.
        modo: "secuencial", "procesos" o "hilos".
        fechas: subconjunto de fechas (por defecto las 11 de `config.FECHAS`).

    Returns:
        (cubo, scl, fechas, perfil) según el contrato del módulo.
    """
    if modo not in MODOS:
        raise ValueError(f"modo debe ser uno de {MODOS}, no {modo!r}")
    fechas = list(fechas or FECHAS[lago])
    alto, ancho = forma_malla(lago)
    forma_cubo = (len(fechas), N_BANDAS, alto, ancho)
    forma_scl = (len(fechas), alto, ancho)

    if modo == "secuencial":
        cubo = np.empty(forma_cubo, dtype=np.float32)
        scl = np.empty(forma_scl, dtype=np.uint8)
        for i, fecha in enumerate(fechas):
            _leer_fecha(lago, fecha, cubo[i], scl[i])

    elif modo == "hilos":
        cubo = np.empty(forma_cubo, dtype=np.float32)
        scl = np.empty(forma_scl, dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=_n_trabajadores(n_procesos, len(fechas))) as pool:
            tareas = [
                pool.submit(_leer_fecha, lago, fecha, cubo[i], scl[i])
                for i, fecha in enumerate(fechas)
            ]
            for tarea in tareas:
                tarea.result()

    else:  # procesos
        bytes_cubo = int(np.prod(forma_cubo)) * np.dtype(np.float32).itemsize
        bytes_scl = int(np.prod(forma_scl))
        shm_cubo = shared_memory.SharedMemory(create=True, size=bytes_cubo)
        shm_scl = shared_memory.SharedMemory(create=True, size=bytes_scl)
        try:
            argumentos = [
                (lago, fecha, i, shm_cubo.name, forma_cubo, shm_scl.name, forma_scl)
                for i, fecha in enumerate(fechas)
            ]
            n = _n_trabajadores(n_procesos, len(fechas))
            with ProcessPoolExecutor(max_workers=n) as pool:
                list(pool.map(_leer_fecha_compartida, argumentos))
            vista_cubo = np.ndarray(forma_cubo, dtype=np.float32, buffer=shm_cubo.buf)
            vista_scl = np.ndarray(forma_scl, dtype=np.uint8, buffer=shm_scl.buf)
            cubo = vista_cubo.copy()
            scl = vista_scl.copy()
            del vista_cubo, vista_scl
        finally:
            shm_cubo.close()
            shm_cubo.unlink()
            shm_scl.close()
            shm_scl.unlink()

    assert cubo.flags["C_CONTIGUOUS"] and cubo.dtype == np.float32
    return cubo, scl, fechas, leer_perfil(lago)


def bandas_de_fecha(cubo: np.ndarray, scl: np.ndarray, i: int) -> dict[str, np.ndarray]:
    """Convierte la fecha `i` del cubo al diccionario que usa `lab4/indices.py`.

    Sirve para correr las funciones originales del laboratorio sobre el cubo
    (validaciones de K1). SCL se devuelve en float32 con NaN donde no hay dato,
    igual que `indices.leer_bandas`.
    """
    bandas = {nombre: cubo[i, b] for b, nombre in enumerate(BANDAS_L1C)}
    capa_scl = scl[i].astype(np.float32)
    capa_scl[scl[i] == SCL_SIN_DATO] = np.nan
    bandas["SCL"] = capa_scl
    return bandas


# --------------------------------------------------------------------------
# Validación y medición
# --------------------------------------------------------------------------
def validar_contra_lab4(lago: str, cubo: np.ndarray, scl: np.ndarray, fechas: list[str]) -> dict:
    """Compara el cubo contra `indices.leer_bandas` del laboratorio, fecha por fecha.

    Exige igualdad exacta (bit a bit, NaN en las mismas posiciones) en las 9 bandas
    y la misma máscara de válidos (`indices.mascara_valida`).
    """
    import indices as ix

    distintos_bandas = 0
    distintos_validos = 0
    for i, fecha in enumerate(fechas):
        ref = ix.leer_bandas(ruta_cruda(lago, fecha, "L1C"), ruta_cruda(lago, fecha, "L2A"))
        for b, nombre in enumerate(BANDAS_L1C):
            if not np.array_equal(cubo[i, b], ref[nombre], equal_nan=True):
                distintos_bandas += 1
        validos_ref = ix.mascara_valida(ref)
        validos_cubo = ix.mascara_valida(bandas_de_fecha(cubo, scl, i))
        distintos_validos += int((validos_ref != validos_cubo).sum())
    return {
        "lago": lago,
        "fechas": len(fechas),
        "forma_cubo": "x".join(map(str, cubo.shape)),
        "megabytes_cubo": round(cubo.nbytes / 1e6, 1),
        "bandas_distintas": distintos_bandas,
        "pixeles_validos_distintos": distintos_validos,
        "ok": distintos_bandas == 0 and distintos_validos == 0,
    }


def _medir_lectura(n_procesos: int | None) -> None:
    import pandas as pd

    from src import RESULTADOS
    from src.medicion.tiempos import guardar, medir

    n = n_procesos or os.cpu_count()
    validaciones = []
    mediciones = []
    for lago in FECHAS:
        cubo, scl, fechas, _ = leer_cubo(lago, modo="secuencial")
        validaciones.append(validar_contra_lab4(lago, cubo, scl, fechas))
        for modo in ("procesos", "hilos"):
            cubo_m, scl_m, _, _ = leer_cubo(lago, n_procesos=n, modo=modo)
            igual = np.array_equal(cubo, cubo_m, equal_nan=True) and np.array_equal(scl, scl_m)
            validaciones[-1][f"{modo}_igual_a_secuencial"] = igual
            validaciones[-1]["ok"] &= igual
            del cubo_m, scl_m
        print(f"[validación] {validaciones[-1]}")
        tam = f"{cubo.nbytes / 1e6:.0f} MB, {len(fechas)} fechas"
        del cubo, scl

        for modo, version in [
            ("secuencial", "CPU secuencial"),
            ("procesos", "CPU multinucleo"),
            ("hilos", "CPU multinucleo"),
        ]:
            r = medir(
                lambda: leer_cubo(lago, n_procesos=n, modo=modo),
                fase=f"Lectura {lago}",
                version=version,
                notas=f"modo={modo}; trabajadores={1 if modo == 'secuencial' else min(n, len(fechas))}; {tam}",
            )
            mediciones.append(r)
            print(f"  {lago:10s} {modo:10s} {r.promedio_s:8.3f} s ± {r.desviacion_s:.3f}")

    guardar(mediciones, "parte1_lectura.csv", sobrescribir=True)
    destino = RESULTADOS / "validacion" / "lectura_vs_lab4.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validaciones).to_csv(destino, index=False)
    print(f"\nGuardado: resultados/tiempos/parte1_lectura.csv y {destino.relative_to(RESULTADOS.parent)}")


def _evidencia(segundos: float, n_procesos: int | None, modo: str) -> None:
    """Repite la lectura multinúcleo para capturar el Administrador de Tareas."""
    from src.medicion.monitor_cpu import registrar_cpu

    print(f"Leyendo en bucle con modo={modo} durante ~{segundos:.0f} s. Abrir el "
          "Administrador de Tareas (Rendimiento > CPU > Procesadores lógicos) y capturar.")
    with registrar_cpu(f"parte1_lectura_{modo}"):
        fin = time.perf_counter() + segundos
        vueltas = 0
        while time.perf_counter() < fin:
            for lago in FECHAS:
                leer_cubo(lago, n_procesos=n_procesos, modo=modo)
            vueltas += 1
    print(f"Listo: {vueltas} vueltas sobre los 2 lagos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--procesos", type=int, default=None, help="trabajadores (defecto: todos)")
    parser.add_argument("--evidencia", type=float, default=0,
                        help="segundos de lectura en bucle para capturas (no mide)")
    parser.add_argument("--modo", choices=MODOS, default="procesos",
                        help="modo usado con --evidencia")
    args = parser.parse_args()
    if args.evidencia:
        _evidencia(args.evidencia, args.procesos, args.modo)
    else:
        _medir_lectura(args.procesos)
