"""Parte I, K1: validación y medición de las cuatro versiones de los índices por píxel.

Orquesta el pipeline separando CPU y GPU:

    1. CPU  preparacion_cpu.lectura.leer_cubo        lee el cubo del lago (no se mide aquí;
                                                     la lectura la mide Javier en lectura.py)
    2.      versiones de K1 sobre el mismo cubo en memoria:
            CPU secuencial   lab4/indices.py (NumPy), el código original del laboratorio
            CPU 1 nucleo     Numba njit, misma matemática que la GPU, sin paralelismo
            CPU multinucleo  Numba njit(parallel=True) con prange
            GPU              calculo_gpu.kernels_indices, modos "simple" y "streams"
    3.      validación de cada versión contra lab4 (fecha por fecha)
    4.      tiempos con el protocolo de medicion/tiempos.py (1 calentamiento + N corridas)

Uso desde la raíz del repositorio:

    .venv\\Scripts\\python -m src.parte1_k1                  # valida y mide todo
    .venv\\Scripts\\python -m src.parte1_k1 --solo-validar   # sin medir tiempos
    .venv\\Scripts\\python -m src.parte1_k1 --sin-gpu        # máquina sin GPU NVIDIA
    .venv\\Scripts\\python -m src.parte1_k1 --evidencia 40   # K1 en GPU en bucle ~40 s
                                                            # (capturas de nvidia-smi)

Salidas:
    resultados/validacion/k1_vs_lab4.csv
    resultados/tiempos/parte1_k1.csv
    evidencias/nvidia-smi/parte1_k1.csv   (registro de nvidia-smi durante las corridas GPU)

Responsable: Nadissa.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import pandas as pd

import src  # noqa: F401  (agrega lab4/ al sys.path)
from config import FECHAS
from src import RESULTADOS
from src.indices_pixel import CAPAS_FLOAT, CAPAS_MASCARA, UMBRAL_CHLA
from src.medicion.tiempos import Medicion, guardar, medir
from src.preparacion_cpu.lectura import bandas_de_fecha, leer_cubo
from src.referencia_cpu.indices_multinucleo import calcular_indices_cpu
from src.referencia_cpu.parte1_secuencial import k1_completo


# --------------------------------------------------------------------------
# Versión secuencial del laboratorio sobre el cubo
# --------------------------------------------------------------------------
def k1_lab4(cubo: np.ndarray, scl: np.ndarray) -> list[dict]:
    """`indices.calcular_indices` (sin lectura) para cada fecha del cubo."""
    return [k1_completo(bandas_de_fecha(cubo, scl, i)) for i in range(cubo.shape[0])]


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def comparar(
    lago: str, version: str, capas: np.ndarray, mascaras: np.ndarray, referencia: list[dict]
) -> list[dict]:
    """Diferencias capa por capa contra lab4, acumuladas sobre todas las fechas."""
    filas = []
    total = capas.shape[0] * capas.shape[2] * capas.shape[3]
    for c, nombre in enumerate(CAPAS_FLOAT):
        maximo, suma, n, nan_distintos = 0.0, 0.0, 0, 0
        for f, ref in enumerate(referencia):
            a, b = capas[f, c], ref[nombre].astype(np.float32)
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            nan_distintos += int((nan_a != nan_b).sum())
            ambos = ~nan_a & ~nan_b
            dif = np.abs(a[ambos].astype(np.float64) - b[ambos].astype(np.float64))
            if dif.size:
                maximo = max(maximo, float(dif.max()))
                suma += float(dif.sum())
                n += dif.size
        filas.append({
            "lago": lago, "version": version, "capa": nombre, "tipo": "float32",
            "error_abs_max": maximo, "error_abs_medio": suma / n if n else 0.0,
            "nan_distintos": nan_distintos, "pixeles_distintos": None,
            "pct_distintos": None, "pixeles_total": total,
        })
    for m, nombre in enumerate(CAPAS_MASCARA):
        distintos = 0
        for f, ref in enumerate(referencia):
            esperado = ref["chla"] > UMBRAL_CHLA if nombre == "y_alta" else ref[nombre]
            distintos += int((mascaras[f, m].astype(bool) != esperado).sum())
        filas.append({
            "lago": lago, "version": version, "capa": nombre, "tipo": "mascara",
            "error_abs_max": None, "error_abs_medio": None, "nan_distintos": None,
            "pixeles_distintos": distintos, "pct_distintos": distintos / total * 100,
            "pixeles_total": total,
        })
    return filas


def resumen_validacion(filas: list[dict]) -> str:
    df = pd.DataFrame(filas)
    flot = df[df["tipo"] == "float32"].groupby("version").agg(
        error_max=("error_abs_max", "max"), nan_distintos=("nan_distintos", "sum"))
    masc = df[df["tipo"] == "mascara"].groupby("version")["pixeles_distintos"].sum()
    return flot.join(masc.rename("mascaras_distintas")).to_string()


# --------------------------------------------------------------------------
# Medición
# --------------------------------------------------------------------------
def _medicion_desde_corridas(fase, version, corridas, notas) -> Medicion:
    return Medicion(
        fase=fase, version=version, promedio_s=statistics.mean(corridas),
        desviacion_s=statistics.stdev(corridas) if len(corridas) > 1 else 0.0,
        corridas_s=list(corridas), notas=notas,
    )


def medir_gpu_por_fase(cubo, scl, lago, repeticiones, nota) -> list[Medicion]:
    """Corre el modo "simple" y reporta cada fase (eventos de CUDA) y el total."""
    from src.calculo_gpu.kernels_indices import calcular_indices_gpu

    calcular_indices_gpu(cubo, scl, "simple")  # calentamiento
    fases = {"copia_cpu_gpu_s": [], "kernel_s": [], "copia_gpu_cpu_s": []}
    totales = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        _, _, t = calcular_indices_gpu(cubo, scl, "simple")
        totales.append(time.perf_counter() - inicio)
        for k in fases:
            fases[k].append(t[k])
    nombres = {"copia_cpu_gpu_s": "copia CPU->GPU", "kernel_s": "kernel",
               "copia_gpu_cpu_s": "copia GPU->CPU"}
    salida = [
        _medicion_desde_corridas(f"K1 {nombres[k]} {lago}", "GPU", v, f"{nota}; eventos CUDA")
        for k, v in fases.items()
    ]
    salida.append(_medicion_desde_corridas(
        f"K1 total con transferencias {lago}", "GPU", totales,
        f"{nota}; modo simple; reloj de pared (incluye reservar memoria en GPU)"))
    return salida


def medir_lago(lago, cubo, scl, repeticiones, usar_gpu, fechas_por_lote) -> list[Medicion]:
    n_f, _, alto, ancho = cubo.shape
    nota = f"{lago}; {n_f} fechas; malla {alto}x{ancho}; cubo {cubo.nbytes / 1e6:.0f} MB"
    mediciones = []

    def fase(nombre, version, funcion, **kw):
        r = medir(funcion, fase=f"{nombre} {lago}", version=version,
                  repeticiones=repeticiones, notas=kw.pop("notas", nota), **kw)
        print(f"  {r.fase:42s} {r.version:16s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
        mediciones.append(r)

    fase("K1", "CPU secuencial", lambda: k1_lab4(cubo, scl), notas=f"{nota}; lab4 NumPy")
    fase("K1", "CPU 1 nucleo", lambda: calcular_indices_cpu(cubo, scl, paralelo=False),
         notas=f"{nota}; Numba njit (fusión sin paralelismo)")
    import numba
    fase("K1", "CPU multinucleo", lambda: calcular_indices_cpu(cubo, scl, paralelo=True),
         notas=f"{nota}; Numba prange, {numba.get_num_threads()} hilos")

    if usar_gpu:
        from src.calculo_gpu.kernels_indices import calcular_indices_gpu

        for r in medir_gpu_por_fase(cubo, scl, lago, repeticiones, nota):
            print(f"  {r.fase:42s} {r.version:16s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
            mediciones.append(r)
        fase("K1 total con streams", "GPU",
             lambda: calcular_indices_gpu(cubo, scl, "streams", fechas_por_lote),
             sincronizar_gpu=True,
             notas=f"{nota}; {fechas_por_lote} fechas por lote; incluye bloquear memoria del host")
    return mediciones


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Parte I, K1: validación y tiempos")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--solo-validar", action="store_true")
    parser.add_argument("--sin-gpu", action="store_true", help="omitir la GPU")
    parser.add_argument("--fechas-por-lote", type=int, default=2)
    parser.add_argument("--evidencia", type=float, default=0,
                        help="segundos de K1 en GPU en bucle para capturas (no mide)")
    parser.add_argument("--lagos", nargs="+", default=list(FECHAS))
    args = parser.parse_args()

    usar_gpu = not args.sin_gpu
    if usar_gpu:
        from numba import cuda
        if not cuda.is_available():
            raise SystemExit("No hay GPU CUDA visible; usar --sin-gpu.")
        from src.calculo_gpu.kernels_indices import calcular_indices_gpu

    if args.evidencia:
        from src.medicion.monitor_gpu import registrar_gpu
        cubos = {lago: leer_cubo(lago, modo="hilos")[:2] for lago in args.lagos}
        print(f"K1 en GPU en bucle ~{args.evidencia:.0f} s: capturar nvidia-smi y el Administrador de Tareas.")
        with registrar_gpu("parte1_k1_evidencia"):
            fin, vueltas = time.perf_counter() + args.evidencia, 0
            while time.perf_counter() < fin:
                for cubo, scl in cubos.values():
                    calcular_indices_gpu(cubo, scl, "simple")
                vueltas += 1
        print(f"Listo: {vueltas} vueltas.")
        return

    validaciones, mediciones = [], []
    for lago in args.lagos:
        print(f"\n=== {lago} ===")
        cubo, scl, fechas, _ = leer_cubo(lago, modo="hilos")
        referencia = k1_lab4(cubo, scl)

        versiones = {
            "CPU 1 nucleo": calcular_indices_cpu(cubo, scl, paralelo=False),
            "CPU multinucleo": calcular_indices_cpu(cubo, scl, paralelo=True),
        }
        if usar_gpu:
            versiones["GPU simple"] = calcular_indices_gpu(cubo, scl, "simple")[:2]
            versiones["GPU streams"] = calcular_indices_gpu(
                cubo, scl, "streams", args.fechas_por_lote)[:2]
        for version, (capas, mascaras) in versiones.items():
            validaciones += comparar(lago, version, capas, mascaras, referencia)
        print(resumen_validacion([v for v in validaciones if v["lago"] == lago]))
        del versiones, referencia

        if not args.solo_validar:
            if usar_gpu:
                from src.medicion.monitor_gpu import registrar_gpu
                with registrar_gpu(f"parte1_k1_{lago}"):
                    mediciones += medir_lago(lago, cubo, scl, args.repeticiones, True,
                                             args.fechas_por_lote)
            else:
                mediciones += medir_lago(lago, cubo, scl, args.repeticiones, False,
                                         args.fechas_por_lote)
        del cubo, scl

    destino = RESULTADOS / "validacion" / "k1_vs_lab4.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validaciones).to_csv(destino, index=False)
    print(f"\nGuardado: {destino.relative_to(RESULTADOS.parent)}")
    if mediciones:
        guardar(mediciones, "parte1_k1.csv", sobrescribir=True)
        print("Guardado: resultados/tiempos/parte1_k1.csv")


if __name__ == "__main__":
    main()
