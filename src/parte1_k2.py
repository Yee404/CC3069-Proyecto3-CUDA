"""Parte I, K2: máscara fija del lago y estadísticas por fecha; pipeline completo de la Parte I.

Valida contra el laboratorio y mide dos cosas:

A. K2 aislado (máscara del lago + estadísticas), con las salidas de K1 ya en memoria:
       CPU secuencial   procesar.construir_mascara_lago + procesar.resumen (lab4, en memoria)
       CPU 1 nucleo     Numba njit
       CPU multinucleo  Numba prange
       GPU              k_mascara_lago + k2_reducciones sobre las capas de K1 que ya están
                        en la GPU; mediana y p90 en CPU sobre los valores compactados

B. Parte I completa (K1 + máscara + K2) desde el cubo en RAM:
       CPU secuencial   lab4 (NumPy)
       CPU multinucleo  Numba prange (K1 + K2)
       GPU, datos en GPU      copia el cubo, K1, máscara, K2 y devuelve solo estadísticas
       GPU, devolviendo capas lo anterior + copia de las capas al CPU (lo que necesita el
                              laboratorio para escribir los GeoTIFF y armar el dataset)

Uso desde la raíz del repositorio:

    .venv\\Scripts\\python -m src.parte1_k2
    .venv\\Scripts\\python -m src.parte1_k2 --solo-validar
    .venv\\Scripts\\python -m src.parte1_k2 --sin-gpu

Salidas:
    resultados/validacion/k2_vs_lab4.csv          (diferencia por columna y versión)
    resultados/validacion/k2_estadisticas_gpu.csv (la tabla que produce la GPU)
    resultados/tiempos/parte1_k2.csv
    evidencias/nvidia-smi/parte1_k2_<lago>.csv

Responsable: Nadissa.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import pandas as pd

import src  # noqa: F401  (agrega lab4/ al sys.path)
import procesar
from config import FECHAS
from src import RESULTADOS
from src.estadisticas import COLUMNAS, armar_tabla, orden_en_cpu
from src.medicion.tiempos import Medicion, guardar, medir
from src.parte1_k1 import k1_lab4
from src.preparacion_cpu.lectura import leer_cubo
from src.referencia_cpu.indices_multinucleo import calcular_indices_cpu
from src.referencia_cpu.parte1_secuencial import mascara_lago
from src.referencia_cpu.reducciones_multinucleo import estadisticas_cpu

NUMERICAS = [c for c in COLUMNAS if c not in ("lago", "fecha")]

# Corridas de calentamiento de las fases GPU. Con 1 sola, en Amatitlán las primeras corridas
# medidas salían ~2.6x más lentas (0.08 vs 0.03 s): la GPU de la laptop baja su frecuencia
# durante las fases de CPU y tarda en volver a subirla. Con 3 ya está estable.
CALENTAMIENTO_GPU = 3


# --------------------------------------------------------------------------
# Versiones
# --------------------------------------------------------------------------
def k2_lab4(lago: str, fechas: list[str], capas_lab: list[dict]) -> tuple[np.ndarray, pd.DataFrame]:
    """Máscara del lago y `procesar.resumen` del laboratorio, en memoria."""
    mascara = mascara_lago(capas_lab)
    tabla = pd.DataFrame([procesar.resumen(lago, f, c, mascara) for f, c in zip(fechas, capas_lab)])
    return mascara, tabla


def k2_cpu(lago, fechas, capas, mascaras, paralelo) -> tuple[np.ndarray, pd.DataFrame]:
    r = estadisticas_cpu(capas, mascaras, paralelo=paralelo)
    n_lago = int(r["lago"].sum())
    return r["lago"], armar_tabla(lago, fechas, n_lago, r["acum"], r["maximo"], r["medianas"], r["p90"])


def k2_gpu(lago, fechas, d_capas, d_mascaras) -> tuple[np.ndarray, pd.DataFrame, dict]:
    from src.calculo_gpu.reducciones import estadisticas_gpu

    r = estadisticas_gpu(d_capas, d_mascaras)
    inicio = time.perf_counter()
    medianas, p90 = orden_en_cpu(r["valores"])
    r["tiempos"]["orden_cpu_s"] = time.perf_counter() - inicio
    n_lago = int(r["lago"].sum())
    tabla = armar_tabla(lago, fechas, n_lago, r["acum"], r["maximo"], medianas, p90)
    return r["lago"], tabla, r["tiempos"]


def parte1_gpu(lago, fechas, cubo, scl, devolver_capas: bool):
    """K1 + máscara + K2 en GPU desde el cubo en RAM."""
    from numba import cuda

    from src.calculo_gpu.kernels_indices import k1_en_dispositivo

    d_cubo = cuda.to_device(cubo)
    d_scl = cuda.to_device(scl)
    d_capas, d_mascaras = k1_en_dispositivo(d_cubo, d_scl)
    del d_cubo, d_scl
    lago_m, tabla, _ = k2_gpu(lago, fechas, d_capas, d_mascaras)
    if devolver_capas:
        return lago_m, tabla, d_capas.copy_to_host(), d_mascaras.copy_to_host()
    return lago_m, tabla


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def comparar(lago, version, mascara, tabla, mascara_ref, tabla_ref) -> list[dict]:
    filas = [{
        "lago": lago, "version": version, "columna": "mascara_lago",
        "dif_abs_max": None, "dif_rel_max": None,
        "pixeles_distintos": int((mascara.astype(bool) != mascara_ref).sum()),
    }]
    for c in NUMERICAS:
        a = tabla[c].to_numpy(dtype=np.float64)
        b = tabla_ref[c].to_numpy(dtype=np.float64)
        dif = np.abs(a - b)
        rel = dif / np.maximum(np.abs(b), 1e-12)
        nan_distintos = int((np.isnan(a) != np.isnan(b)).sum())
        filas.append({
            "lago": lago, "version": version, "columna": c,
            "dif_abs_max": float(np.nanmax(dif)) if np.isfinite(dif).any() else 0.0,
            "dif_rel_max": float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0,
            "pixeles_distintos": nan_distintos or None,
        })
    return filas


def resumen_validacion(filas: list[dict]) -> str:
    df = pd.DataFrame(filas)
    masc = df[df["columna"] == "mascara_lago"].set_index("version")["pixeles_distintos"]
    num = df[df["columna"] != "mascara_lago"].groupby("version")["dif_rel_max"].max()
    exactas = df[df["columna"] == "pixeles_analizados"].set_index("version")["dif_abs_max"]
    return pd.DataFrame({
        "mascara_distinta": masc, "pixeles_analizados_dif": exactas, "dif_rel_max": num,
    }).to_string()


# --------------------------------------------------------------------------
# Medición
# --------------------------------------------------------------------------
def _de_corridas(fase, version, corridas, notas) -> Medicion:
    return Medicion(fase=fase, version=version, promedio_s=statistics.mean(corridas),
                    desviacion_s=statistics.stdev(corridas) if len(corridas) > 1 else 0.0,
                    corridas_s=list(corridas), notas=notas)


def medir_lago(lago, fechas, cubo, scl, repeticiones, usar_gpu) -> list[Medicion]:
    n_f, _, alto, ancho = cubo.shape
    nota = f"{lago}; {n_f} fechas; malla {alto}x{ancho}"
    mediciones = []

    def fase(nombre, version, funcion, notas, **kw):
        r = medir(funcion, fase=f"{nombre} {lago}", version=version,
                  repeticiones=repeticiones, notas=f"{nota}; {notas}", **kw)
        print(f"  {r.fase:46s} {r.version:16s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
        mediciones.append(r)

    capas_lab = k1_lab4(cubo, scl)
    capas, mascaras = calcular_indices_cpu(cubo, scl, paralelo=True)

    # A. K2 aislado
    fase("K2", "CPU secuencial", lambda: k2_lab4(lago, fechas, capas_lab), "lab4 en memoria")
    fase("K2", "CPU 1 nucleo", lambda: k2_cpu(lago, fechas, capas, mascaras, False), "Numba njit")
    fase("K2", "CPU multinucleo", lambda: k2_cpu(lago, fechas, capas, mascaras, True), "Numba prange")
    if usar_gpu:
        from numba import cuda

        d_capas, d_mascaras = cuda.to_device(capas), cuda.to_device(mascaras)
        for _ in range(CALENTAMIENTO_GPU):
            k2_gpu(lago, fechas, d_capas, d_mascaras)
        por_fase = {"mascara_s": [], "k2_s": [], "copia_gpu_cpu_s": [], "orden_cpu_s": []}
        totales = []
        for _ in range(repeticiones):
            inicio = time.perf_counter()
            _, _, t = k2_gpu(lago, fechas, d_capas, d_mascaras)
            totales.append(time.perf_counter() - inicio)
            for k in por_fase:
                por_fase[k].append(t[k])
        nombres = {"mascara_s": "K2 kernel mascara del lago", "k2_s": "K2 kernel reducciones",
                   "copia_gpu_cpu_s": "K2 copia GPU->CPU (resultados)",
                   "orden_cpu_s": "K2 mediana y p90 en CPU"}
        for k, v in por_fase.items():
            r = _de_corridas(f"{nombres[k]} {lago}", "GPU", v, f"{nota}; capas de K1 ya en GPU")
            print(f"  {r.fase:46s} {r.version:16s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
            mediciones.append(r)
        r = _de_corridas(f"K2 {lago}", "GPU", totales,
                         f"{nota}; capas de K1 ya en GPU; incluye mediana/p90 en CPU y armar tabla")
        print(f"  {r.fase:46s} {r.version:16s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
        mediciones.append(r)
        del d_capas, d_mascaras

    # B. Parte I completa desde el cubo en RAM
    fase("Parte I (K1+K2)", "CPU secuencial",
         lambda: k2_lab4(lago, fechas, k1_lab4(cubo, scl)), "lab4 NumPy")
    fase("Parte I (K1+K2)", "CPU multinucleo",
         lambda: k2_cpu(lago, fechas, *calcular_indices_cpu(cubo, scl, True), True), "Numba prange")
    if usar_gpu:
        fase("Parte I (K1+K2) datos en GPU", "GPU",
             lambda: parte1_gpu(lago, fechas, cubo, scl, False),
             "copia el cubo, K1, mascara, K2; devuelve solo estadisticas", sincronizar_gpu=True,
             calentamiento=CALENTAMIENTO_GPU)
        fase("Parte I (K1+K2) devolviendo capas", "GPU",
             lambda: parte1_gpu(lago, fechas, cubo, scl, True),
             "igual + copia de capas y mascaras al CPU", sincronizar_gpu=True,
             calentamiento=CALENTAMIENTO_GPU)
    return mediciones


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Parte I, K2: validación y tiempos")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--solo-validar", action="store_true")
    parser.add_argument("--sin-gpu", action="store_true")
    parser.add_argument("--lagos", nargs="+", default=list(FECHAS))
    args = parser.parse_args()

    usar_gpu = not args.sin_gpu
    if usar_gpu:
        from numba import cuda
        if not cuda.is_available():
            raise SystemExit("No hay GPU CUDA visible; usar --sin-gpu.")

    validaciones, mediciones, tablas_gpu = [], [], []
    for lago in args.lagos:
        print(f"\n=== {lago} ===")
        cubo, scl, fechas, _ = leer_cubo(lago, modo="hilos")
        mascara_ref, tabla_ref = k2_lab4(lago, fechas, k1_lab4(cubo, scl))
        capas, mascaras = calcular_indices_cpu(cubo, scl, paralelo=True)

        for version, paralelo in (("CPU 1 nucleo", False), ("CPU multinucleo", True)):
            m, t = k2_cpu(lago, fechas, capas, mascaras, paralelo)
            validaciones += comparar(lago, version, m, t, mascara_ref, tabla_ref)
        if usar_gpu:
            m, t = parte1_gpu(lago, fechas, cubo, scl, False)
            validaciones += comparar(lago, "GPU", m, t, mascara_ref, tabla_ref)
            tablas_gpu.append(t)
        print(resumen_validacion([v for v in validaciones if v["lago"] == lago]))
        del capas, mascaras

        if not args.solo_validar:
            if usar_gpu:
                from src.medicion.monitor_gpu import registrar_gpu
                with registrar_gpu(f"parte1_k2_{lago}"):
                    mediciones += medir_lago(lago, fechas, cubo, scl, args.repeticiones, True)
            else:
                mediciones += medir_lago(lago, fechas, cubo, scl, args.repeticiones, False)
        del cubo, scl

    destino = RESULTADOS / "validacion"
    destino.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validaciones).to_csv(destino / "k2_vs_lab4.csv", index=False)
    if tablas_gpu:
        pd.concat(tablas_gpu).to_csv(destino / "k2_estadisticas_gpu.csv", index=False)
    print("\nGuardado: resultados/validacion/k2_vs_lab4.csv")
    if mediciones:
        guardar(mediciones, "parte1_k2.csv", sobrescribir=True)
        print("Guardado: resultados/tiempos/parte1_k2.csv")


if __name__ == "__main__":
    main()
