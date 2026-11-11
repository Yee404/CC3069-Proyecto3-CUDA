"""Verifica que el entorno reconoce la GPU y las bibliotecas CUDA.

    python -m src.verificar_entorno

Guardar la salida en evidencias/ como primera evidencia de hardware.
"""

from __future__ import annotations

import platform
import subprocess


def _linea(nombre: str, valor: object) -> None:
    print(f"{nombre:<22}{valor}")


def main() -> None:
    _linea("Python", platform.python_version())
    _linea("Sistema", f"{platform.system()} {platform.release()}")

    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        _linea("nvidia-smi", salida)
    except (OSError, subprocess.CalledProcessError) as error:
        _linea("nvidia-smi", f"NO DISPONIBLE ({error})")

    try:
        import numpy as np
        from numba import __version__ as v_numba
        from numba import cuda

        _linea("numba", v_numba)
        _linea("cuda.is_available()", cuda.is_available())
        if cuda.is_available():
            dispositivo = cuda.get_current_device()
            _linea("GPU (numba)", dispositivo.name.decode() if isinstance(dispositivo.name, bytes) else dispositivo.name)
            _linea("Compute capability", dispositivo.compute_capability)

            @cuda.jit
            def suma(a, b, c):
                i = cuda.grid(1)
                if i < c.size:
                    c[i] = a[i] + b[i]

            n = 1_000_000
            a = np.ones(n, dtype=np.float32)
            c = cuda.device_array_like(a)
            suma[(n + 255) // 256, 256](cuda.to_device(a), cuda.to_device(a), c)
            _linea("Kernel de prueba", "OK" if np.allclose(c.copy_to_host(), 2.0) else "FALLO")
    except Exception as error:  # noqa: BLE001
        _linea("numba-cuda", f"ERROR: {error}")

    try:
        import cupy as cp

        _linea("cupy", f"{cp.__version__} ({cp.cuda.runtime.getDeviceCount()} GPU)")
    except Exception as error:  # noqa: BLE001
        _linea("cupy", f"ERROR: {error}")

    try:
        import warnings

        import numpy as np
        import xgboost as xgb

        X = np.random.default_rng(0).random((1000, 5), dtype=np.float32)
        y = (X[:, 0] > 0.5).astype(np.int8)
        # Sin GPU visible, XGBoost no falla: avisa y entrena en CPU. Se capturan los
        # avisos para no reportar un "OK" falso.
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            xgb.XGBClassifier(n_estimators=5, tree_method="hist", device="cuda").fit(X, y)
        cayo_a_cpu = any("GPU" in str(a.message) for a in avisos)
        estado = "CAYO A CPU: no hay GPU visible" if cayo_a_cpu else "device='cuda' OK"
        _linea("xgboost", f"{xgb.__version__} ({estado})")
    except Exception as error:  # noqa: BLE001
        _linea("xgboost", f"ERROR: {error}")


if __name__ == "__main__":
    main()
