"""Protocolo de medición común para todas las fases (PLAN.md §7).

Uso:

    from src.medicion.tiempos import medir, guardar

    r = medir(lambda: calcular_indices_gpu(cubo, scl), fase="K1 computo",
              version="GPU", sincronizar_gpu=True)
    guardar([r], "parte1_k1.csv")

Una corrida de calentamiento (compilación JIT) se descarta y luego se hacen
`repeticiones` corridas; se reporta promedio, desviación estándar y cada corrida.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd

from src import RESULTADOS


@dataclass
class Medicion:
    fase: str
    version: str  # "CPU secuencial" | "CPU multinucleo" | "GPU"
    promedio_s: float
    desviacion_s: float
    corridas_s: list[float] = field(default_factory=list)
    notas: str = ""


def _sincronizar() -> None:
    from numba import cuda

    cuda.synchronize()


def medir(
    funcion: Callable[[], object],
    fase: str,
    version: str,
    repeticiones: int = 5,
    calentamiento: int = 1,
    sincronizar_gpu: bool = False,
    notas: str = "",
) -> Medicion:
    """Ejecuta `funcion` y mide su tiempo de pared con time.perf_counter.

    Con `sincronizar_gpu=True` se llama a cuda.synchronize() antes de parar el
    reloj, porque los kernels de CUDA se lanzan de forma asíncrona.
    """
    for _ in range(calentamiento):
        funcion()
        if sincronizar_gpu:
            _sincronizar()

    corridas = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion()
        if sincronizar_gpu:
            _sincronizar()
        corridas.append(time.perf_counter() - inicio)

    return Medicion(
        fase=fase,
        version=version,
        promedio_s=statistics.mean(corridas),
        desviacion_s=statistics.stdev(corridas) if len(corridas) > 1 else 0.0,
        corridas_s=corridas,
        notas=notas,
    )


def guardar(
    mediciones: list[Medicion], nombre_archivo: str, sobrescribir: bool = False
) -> pd.DataFrame:
    """Agrega las mediciones a resultados/tiempos/<nombre_archivo> (CSV).

    Con `sobrescribir=True` el archivo se reemplaza en lugar de agregar filas, para
    que volver a correr un script no duplique las mediciones.
    """
    destino = RESULTADOS / "tiempos" / nombre_archivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame([asdict(m) for m in mediciones])
    tabla["corridas_s"] = tabla["corridas_s"].apply(lambda c: ";".join(f"{v:.6f}" for v in c))
    if sobrescribir:
        tabla.to_csv(destino, index=False)
    else:
        tabla.to_csv(destino, mode="a", header=not destino.exists(), index=False)
    return tabla


def speedup(t_secuencial: float, t_paralelo: float) -> float:
    """S = T_secuencial / T_paralelo."""
    return t_secuencial / t_paralelo
