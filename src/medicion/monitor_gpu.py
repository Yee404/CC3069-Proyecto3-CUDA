"""Registro de nvidia-smi durante una corrida (evidencia para el informe).

    from src.medicion.monitor_gpu import registrar_gpu

    with registrar_gpu("parte2_xgboost_gpu"):
        entrenar_xgboost_gpu(...)

Escribe evidencias/nvidia-smi/<nombre>.csv con una muestra cada `intervalo_ms`.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager

from src import EVIDENCIAS

CAMPOS = (
    "timestamp,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,temperature.gpu,power.draw"
)


@contextmanager
def registrar_gpu(nombre: str, intervalo_ms: int = 200):
    destino = EVIDENCIAS / "nvidia-smi" / f"{nombre}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    proceso = subprocess.Popen(
        [
            "nvidia-smi",
            f"--query-gpu={CAMPOS}",
            "--format=csv",
            f"-lms={intervalo_ms}",
            f"--filename={destino}",
        ]
    )
    try:
        yield destino
    finally:
        proceso.terminate()
        proceso.wait(timeout=5)
