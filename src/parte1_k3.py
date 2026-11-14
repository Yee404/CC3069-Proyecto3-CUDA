"""Parte I, K3: histogramas de clorofila-a por fecha (notebook 02, celda 48).

Valida y mide K3 aislado, con las capas de K1 y la máscara del lago ya en memoria
(o en la GPU):

    CPU secuencial   como el notebook: por fecha, filtrar con NumPy, recortar y np.histogram
    CPU 1 nucleo     Numba njit
    CPU multinucleo  Numba prange (una fecha por núcleo)
    GPU              k3_histograma: histograma privado por bloque en memoria compartida

Además dibuja la figura del notebook 02 con los histogramas de la GPU (o de NumPy con --sin-gpu)
(`resultados/graficos/k3_histogramas.png`), como evidencia de que el resultado es el mismo.

Uso desde la raíz del repositorio:

    .venv\\Scripts\\python -m src.parte1_k3
    .venv\\Scripts\\python -m src.parte1_k3 --solo-validar
    .venv\\Scripts\\python -m src.parte1_k3 --sin-gpu

Salidas:
    resultados/validacion/k3_vs_lab4.csv
    resultados/validacion/k3_histogramas.csv    (conteos por lago, fecha y bin)
    resultados/tiempos/parte1_k3.csv
    resultados/graficos/k3_histogramas.png

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
from src.estadisticas import LIMITE_HISTOGRAMA, N_BINS, bordes_histograma, histogramas_lab
from src.medicion.tiempos import Medicion, guardar, medir
from src.parte1_k1 import k1_lab4
from src.preparacion_cpu.lectura import leer_cubo
from src.referencia_cpu.indices_multinucleo import calcular_indices_cpu
from src.referencia_cpu.parte1_secuencial import mascara_lago
from src.referencia_cpu.reducciones_multinucleo import histogramas_cpu, mascara_lago_cpu

CALENTAMIENTO_GPU = 3


def comparar(lago, version, hist, hist_ref) -> dict:
    return {
        "lago": lago, "version": version,
        "conteos_distintos": int((hist != hist_ref).sum()),
        "dif_abs_max": int(np.abs(hist - hist_ref).max()),
        "total": int(hist.sum()), "total_ref": int(hist_ref.sum()),
    }


def tabla_conteos(lago, fechas, hist, bordes) -> pd.DataFrame:
    filas = []
    for f, fecha in enumerate(fechas):
        for b in range(N_BINS):
            filas.append({"lago": lago, "fecha": fecha, "bin": b,
                          "desde": float(bordes[b]), "hasta": float(bordes[b + 1]),
                          "conteo": int(hist[f, b])})
    return pd.DataFrame(filas)


def figura(histos: dict, destino, origen: str) -> None:
    """Misma figura que el notebook 02 (densidad por fecha); `origen` dice de dónde salen los conteos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, len(histos), figsize=(8.5 * len(histos), 5))
    for eje, (lago, (fechas, hist, bordes)) in zip(np.atleast_1d(ejes), histos.items()):
        ancho = np.diff(bordes)
        for f, fecha in enumerate(fechas):
            densidad = hist[f] / (hist[f].sum() * ancho)
            eje.stairs(densidad, bordes, label=fecha, linewidth=1.3)
        eje.set_title(f"Lago de {lago} — distribución de clorofila-a por fecha ({origen})")
        eje.set_xlabel("Clorofila-a (mg/m3)")
        eje.set_ylabel("Densidad")
        eje.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def medir_lago(lago, capas_lab, mascara_ref, capas, mascaras, lago_m, bordes,
               repeticiones, usar_gpu) -> list[Medicion]:
    n_f, _, alto, ancho = capas.shape
    nota = f"{lago}; {n_f} fechas; malla {alto}x{ancho}; {N_BINS} bins en [0, {LIMITE_HISTOGRAMA[lago]}]"
    mediciones = []

    def fase(version, funcion, notas):
        r = medir(funcion, fase=f"K3 {lago}", version=version, repeticiones=repeticiones,
                  notas=f"{nota}; {notas}")
        print(f"  {r.fase:40s} {r.version:16s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
        mediciones.append(r)

    fase("CPU secuencial", lambda: histogramas_lab(lago, capas_lab, mascara_ref), "NumPy como el notebook 02")
    fase("CPU 1 nucleo", lambda: histogramas_cpu(capas, mascaras, lago_m, bordes, False), "Numba njit")
    fase("CPU multinucleo", lambda: histogramas_cpu(capas, mascaras, lago_m, bordes, True), "Numba prange")

    if usar_gpu:
        from numba import cuda

        from src.calculo_gpu.reducciones import histogramas_gpu

        d_capas, d_mascaras, d_lago = (cuda.to_device(capas), cuda.to_device(mascaras),
                                       cuda.to_device(lago_m))
        for _ in range(CALENTAMIENTO_GPU):
            histogramas_gpu(d_capas, d_mascaras, d_lago, bordes)
        por_fase = {"k3_s": [], "copia_gpu_cpu_s": []}
        totales = []
        for _ in range(repeticiones):
            inicio = time.perf_counter()
            _, t = histogramas_gpu(d_capas, d_mascaras, d_lago, bordes)
            totales.append(time.perf_counter() - inicio)
            for k in por_fase:
                por_fase[k].append(t[k])
        nombres = {"k3_s": "K3 kernel", "copia_gpu_cpu_s": "K3 copia GPU->CPU (histogramas)"}
        for k, v in list(por_fase.items()) + [("total", totales)]:
            nombre = nombres.get(k, "K3")
            r = Medicion(fase=f"{nombre} {lago}", version="GPU", promedio_s=statistics.mean(v),
                         desviacion_s=statistics.stdev(v) if len(v) > 1 else 0.0,
                         corridas_s=list(v), notas=f"{nota}; capas y máscara ya en GPU")
            print(f"  {r.fase:40s} {r.version:16s} {r.promedio_s:8.4f} s ± {r.desviacion_s:.4f}")
            mediciones.append(r)
    return mediciones


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte I, K3: validación y tiempos")
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

    validaciones, mediciones, conteos, histos = [], [], [], {}
    for lago in args.lagos:
        print(f"\n=== {lago} ===")
        cubo, scl, fechas, _ = leer_cubo(lago, modo="hilos")
        bordes = bordes_histograma(lago)
        capas_lab = k1_lab4(cubo, scl)
        mascara_ref = mascara_lago(capas_lab)
        hist_ref = histogramas_lab(lago, capas_lab, mascara_ref)

        capas, mascaras = calcular_indices_cpu(cubo, scl, paralelo=True)
        lago_m = mascara_lago_cpu(mascaras)
        for version, paralelo in (("CPU 1 nucleo", False), ("CPU multinucleo", True)):
            h = histogramas_cpu(capas, mascaras, lago_m, bordes, paralelo)
            validaciones.append(comparar(lago, version, h, hist_ref))
        hist_figura = hist_ref
        if usar_gpu:
            from numba import cuda

            from src.calculo_gpu.kernels_indices import k1_en_dispositivo
            from src.calculo_gpu.reducciones import histogramas_gpu, mascara_lago_gpu

            d_capas, d_mascaras = k1_en_dispositivo(cuda.to_device(cubo), cuda.to_device(scl))
            h, _ = histogramas_gpu(d_capas, d_mascaras, mascara_lago_gpu(d_mascaras), bordes)
            validaciones.append(comparar(lago, "GPU", h, hist_ref))
            hist_figura = h
            del d_capas, d_mascaras
        conteos.append(tabla_conteos(lago, fechas, hist_figura, bordes))
        histos[lago] = (fechas, hist_figura, bordes)
        print(pd.DataFrame([v for v in validaciones if v["lago"] == lago])
              .set_index("version").drop(columns="lago").to_string())

        if not args.solo_validar:
            if usar_gpu:
                from src.medicion.monitor_gpu import registrar_gpu
                with registrar_gpu(f"parte1_k3_{lago}"):
                    mediciones += medir_lago(lago, capas_lab, mascara_ref, capas, mascaras,
                                             lago_m, bordes, args.repeticiones, True)
            else:
                mediciones += medir_lago(lago, capas_lab, mascara_ref, capas, mascaras,
                                         lago_m, bordes, args.repeticiones, False)
        del cubo, scl, capas_lab, capas, mascaras

    destino = RESULTADOS / "validacion"
    destino.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validaciones).to_csv(destino / "k3_vs_lab4.csv", index=False)
    pd.concat(conteos).to_csv(destino / "k3_histogramas.csv", index=False)
    figura(histos, RESULTADOS / "graficos" / "k3_histogramas.png",
           "K3 en GPU" if usar_gpu else "NumPy, sin GPU")
    print("\nGuardado: resultados/validacion/k3_*.csv y resultados/graficos/k3_histogramas.png")
    if mediciones:
        guardar(mediciones, "parte1_k3.csv", sobrescribir=True)
        print("Guardado: resultados/tiempos/parte1_k3.csv")


if __name__ == "__main__":
    main()
