"""Registro de nvidia-smi durante una corrida (evidencia para el informe).

    from src.medicion.monitor_gpu import registrar_gpu

    with registrar_gpu("parte2_xgboost_gpu"):
        entrenar_xgboost_gpu(...)

Escribe evidencias/nvidia-smi/<nombre>.csv con una muestra cada `intervalo_ms` (o lo que
tarde nvidia-smi en responder, si es más).

Cómo muestrea: un hilo de Python llama a `nvidia-smi --query-gpu=...` una vez por muestra y
escribe cada línea al archivo en cuanto llega (con flush). La versión anterior dejaba
corriendo `nvidia-smi -lms ... --filename=...`, que escribe en bloques de 4 KB; al terminar
la corrida, `terminate()` lo mataba en Windows antes de vaciar el último bloque, y las
corridas cortas quedaban vacías o cortadas (registros de la Parte I del 2026-11-15).

El formato es el mismo que el de nvidia-smi con `--format=csv` (encabezado con unidades y
marca de tiempo propia de nvidia-smi), así que los CSV anteriores y los nuevos se leen igual.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from contextlib import contextmanager

from src import EVIDENCIAS

CAMPOS = (
    "timestamp,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,temperature.gpu,power.draw"
)


def _consultar(con_encabezado: bool) -> list[str]:
    """Una muestra de nvidia-smi (con encabezado la primera vez)."""
    formato = "csv" if con_encabezado else "csv,noheader"
    salida = subprocess.run(
        ["nvidia-smi", f"--query-gpu={CAMPOS}", f"--format={formato}"],
        capture_output=True, text=True, timeout=10,
    )
    return [linea for linea in salida.stdout.splitlines() if linea.strip()]


@contextmanager
def registrar_gpu(nombre: str, intervalo_ms: int = 200):
    if shutil.which("nvidia-smi") is None:
        raise FileNotFoundError("nvidia-smi no está disponible (¿hay GPU NVIDIA y driver?)")
    destino = EVIDENCIAS / "nvidia-smi" / f"{nombre}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    detener = threading.Event()

    def _muestrear() -> None:
        with open(destino, "w", encoding="utf-8", newline="") as archivo:
            falta_encabezado = True
            while True:
                inicio = time.perf_counter()
                try:
                    lineas = _consultar(falta_encabezado)
                except (OSError, subprocess.SubprocessError):
                    lineas = []  # una muestra perdida no detiene el registro
                for linea in lineas:
                    archivo.write(linea + "\n")
                archivo.flush()
                if lineas:
                    falta_encabezado = False
                espera = intervalo_ms / 1000 - (time.perf_counter() - inicio)
                if detener.wait(max(0.0, espera)):
                    break

    hilo = threading.Thread(target=_muestrear, name=f"nvidia-smi {nombre}", daemon=True)
    hilo.start()
    try:
        yield destino
    finally:
        detener.set()
        hilo.join(timeout=15)
