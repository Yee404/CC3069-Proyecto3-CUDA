"""Registro del uso de CPU por núcleo durante una corrida (evidencia para el informe).

Complementa las capturas del Administrador de Tareas con un log numérico:

    from src.medicion.monitor_cpu import registrar_cpu

    with registrar_cpu("parte1_lectura_procesos"):
        leer_cubo("Atitlan", modo="procesos")

Escribe evidencias/administrador-tareas/<nombre>_cpu.csv con una fila cada
`intervalo_s`: marca de tiempo, uso total y uso de cada núcleo lógico (%), y RAM usada.
"""

from __future__ import annotations

import csv
import threading
import time
from contextlib import contextmanager

import psutil

from src import EVIDENCIAS


@contextmanager
def registrar_cpu(nombre: str, intervalo_s: float = 0.25):
    destino = EVIDENCIAS / "administrador-tareas" / f"{nombre}_cpu.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    n_nucleos = psutil.cpu_count(logical=True)
    detener = threading.Event()

    def _muestrear() -> None:
        with open(destino, "w", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(
                ["t_s", "cpu_total_pct"]
                + [f"nucleo_{i}_pct" for i in range(n_nucleos)]
                + ["ram_usada_gb"]
            )
            psutil.cpu_percent(percpu=True)  # la primera lectura siempre es 0
            inicio = time.perf_counter()
            while not detener.wait(intervalo_s):
                por_nucleo = psutil.cpu_percent(percpu=True)
                escritor.writerow(
                    [f"{time.perf_counter() - inicio:.2f}",
                     f"{sum(por_nucleo) / len(por_nucleo):.1f}"]
                    + [f"{v:.1f}" for v in por_nucleo]
                    + [f"{psutil.virtual_memory().used / 1e9:.2f}"]
                )

    hilo = threading.Thread(target=_muestrear, daemon=True)
    hilo.start()
    try:
        yield destino
    finally:
        detener.set()
        hilo.join(timeout=5)
