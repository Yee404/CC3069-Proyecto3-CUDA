"""Ficha de hardware y software de la máquina donde se mide (PLAN.md §7).

    .venv\\Scripts\\python -m src.medicion.maquina javier

Escribe resultados/maquinas/<integrante>.json con CPU, núcleos, RAM, GPU, sistema
operativo y versiones de las bibliotecas. Cada integrante la genera en su máquina
y la versiona, para que cada tabla de tiempos se pueda atribuir a un hardware.
No guarda el nombre del equipo ni del usuario.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata

import psutil

from src import RESULTADOS

BIBLIOTECAS = [
    "numpy", "pandas", "scipy", "rasterio", "scikit-learn", "xgboost", "shap",
    "numba", "numba-cuda", "cupy-cuda12x",
]


def _cpu() -> str:
    if platform.system() == "Windows":
        try:
            salida = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if salida:
                return salida
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or "desconocido"


def _gpu() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "sin GPU NVIDIA (nvidia-smi no disponible)"


def ficha() -> dict:
    versiones = {}
    for nombre in BIBLIOTECAS:
        try:
            versiones[nombre] = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            versiones[nombre] = None
    return {
        "cpu": _cpu(),
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpu": _gpu(),
        "sistema": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "bibliotecas": versiones,
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python -m src.medicion.maquina <integrante>")
    destino = RESULTADOS / "maquinas" / f"{sys.argv[1].lower()}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    datos = ficha()
    destino.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(datos, indent=2, ensure_ascii=False))
    print(f"\nGuardado: {destino.relative_to(RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
