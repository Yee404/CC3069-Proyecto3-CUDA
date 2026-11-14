"""Corrida corta de la Parte I en GPU para perfilar con Nsight Systems.

No mide ni valida: ejecuta pocas veces cada variante de GPU sobre un lago, marcando cada
una con un rango NVTX para que en la línea de tiempo de Nsight se vea con nombre:

    K1 simple            copia del cubo -> kernel -> copia de capas (fases en serie)
    K1 streams           lotes de fechas en streams: copias y kernels solapados
    Parte I datos en GPU copia del cubo -> K1 -> máscara -> K2 -> K3 -> estadísticas

Los rangos NVTX necesitan el paquete `nvtx` (`pip install nvtx`); sin él el script corre
igual, pero la línea de tiempo no tiene nombres.

Uso (Nsight Systems instalado; ver CONTEXTO_INFORME.md §10.7):

    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true ^
        -o evidencias\\nsight\\parte1_atitlan .venv\\Scripts\\python.exe -m src.perfil_nsight

Responsable: Nadissa.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import src  # noqa: F401  (agrega lab4/ al sys.path)
from src.preparacion_cpu.lectura import leer_cubo

try:
    import nvtx

    def rango(nombre: str, color: str):
        return nvtx.annotate(nombre, color=color)
except ImportError:  # sin nvtx: rangos vacíos
    def rango(nombre: str, color: str):
        return nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parte I en GPU para Nsight Systems")
    parser.add_argument("--lago", default="Atitlan")
    parser.add_argument("--vueltas", type=int, default=3)
    parser.add_argument("--fechas-por-lote", type=int, default=2)
    args = parser.parse_args()

    from numba import cuda

    from src.calculo_gpu.kernels_indices import calcular_indices_gpu, k1_en_dispositivo
    from src.calculo_gpu.reducciones import estadisticas_gpu, histogramas_gpu
    from src.estadisticas import bordes_histograma

    with rango("lectura del cubo (CPU)", "gray"):
        cubo, scl, fechas, _ = leer_cubo(args.lago, modo="hilos")
    bordes = bordes_histograma(args.lago)

    with rango("calentamiento (compilación JIT)", "gray"):
        calcular_indices_gpu(cubo, scl, "simple")
        calcular_indices_gpu(cubo, scl, "streams", args.fechas_por_lote)
        d_c, d_m = k1_en_dispositivo(cuda.to_device(cubo), cuda.to_device(scl))
        r = estadisticas_gpu(d_c, d_m)
        histogramas_gpu(d_c, d_m, r["d_lago"], bordes)
        del d_c, d_m, r
        cuda.synchronize()

    for i in range(args.vueltas):
        with rango(f"K1 simple #{i + 1}", "blue"):
            calcular_indices_gpu(cubo, scl, "simple")
            cuda.synchronize()
    for i in range(args.vueltas):
        with rango(f"K1 streams ({args.fechas_por_lote} fechas por lote) #{i + 1}", "green"):
            calcular_indices_gpu(cubo, scl, "streams", args.fechas_por_lote)
            cuda.synchronize()
    for i in range(args.vueltas):
        with rango(f"Parte I datos en GPU #{i + 1}", "orange"):
            with rango("copia del cubo CPU->GPU", "red"):
                d_cubo, d_scl = cuda.to_device(cubo), cuda.to_device(scl)
            with rango("K1", "blue"):
                d_c, d_m = k1_en_dispositivo(d_cubo, d_scl)
            with rango("mascara del lago + K2", "purple"):
                r = estadisticas_gpu(d_c, d_m)
            with rango("K3", "yellow"):
                histogramas_gpu(d_c, d_m, r["d_lago"], bordes)
            cuda.synchronize()
            del d_cubo, d_scl, d_c, d_m, r
    print(f"Listo: {args.lago}, {args.vueltas} vueltas de cada variante.")


if __name__ == "__main__":
    main()
