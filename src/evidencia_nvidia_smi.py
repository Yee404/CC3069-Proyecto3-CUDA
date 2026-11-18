"""Evidencia de uso de la GPU en la Parte II y medida de las copias CPU<->GPU. María José Yee.

Dos modos (desde la raíz del repositorio); NO modifica `protocolo_parte2.py`:

1) Pasada de evidencia con nvidia-smi (para el informe, 30 pts):

    .venv\\Scripts\\python -m src.evidencia_nvidia_smi --modelo regresion_logistica
    .venv\\Scripts\\python -m src.evidencia_nvidia_smi --modelo xgboost

   Envuelve el protocolo completo de GPU con `registrar_gpu` y guarda
       evidencias/nvidia-smi/parte2_<modelo>_gpu.csv         una muestra cada --intervalo-ms
       evidencias/nvidia-smi/parte2_<modelo>_gpu_fases.csv   cuándo empezó y terminó cada fase
   (las marcas de tiempo usan el mismo formato que el CSV de nvidia-smi, para ubicar en la
   curva el calentamiento y cada vuelta).
   Es una pasada APARTE de la medida: no guarda tiempos ni métricas (los CSV de
   resultados/ quedan intactos) porque muestrear nvidia-smi agrega algo de carga.
   `--vueltas` repite el protocolo para que la grabación del Administrador de Tareas dure
   lo suficiente (por defecto 15 para la Regresión Logística, ~50 s; 1 para XGBoost, ~90 s).
   `--espera` da segundos antes de empezar, para iniciar la grabación y enfocar la ventana.

2) Copias CPU<->GPU (para la tabla de speedup "con y sin copias"):

    .venv\\Scripts\\python -m src.evidencia_nvidia_smi --copias

   Mide con `medicion/tiempos.py` (calentamiento + repeticiones, cuda.synchronize) las
   copias de la unidad representativa de ej4: X,y de entrenamiento a la GPU, X de prueba a
   la GPU y las probabilidades de vuelta al host (float64 en Regresión Logística, float32
   en XGBoost). Salida: resultados/tiempos/parte2_copias_gpu.csv
   (Para XGBoost, `diagnostico_xgboost_gpu.py` ya da además QuantileDMatrix, entrenamiento
   y predicción por separado.)

Este script lee archivos, por eso vive fuera de `calculo_gpu/`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src import EVIDENCIAS
from src.medicion.monitor_gpu import registrar_gpu
from src.medicion.tiempos import guardar, medir
from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones

VUELTAS_POR_DEFECTO = {"regresion_logistica": 15, "xgboost": 1}

# Avisos para las capturas del Administrador de Tareas: (segundos desde que termina la
# cuenta regresiva, mensaje). Se escriben en la consola real (sys.__stdout__) porque durante
# cada vuelta `sys.stdout` está redirigido para no imprimir la línea de cada experimento.
AVISOS_CAPTURA = [
    (35, "Prepárate: en 5 s, captura 1 (Win+Shift+S) de la pantalla DETALLES con la fila de python.exe"),
    (40, ">>> AHORA: captura 1 = pantalla DETALLES (fila de python.exe y su memoria de GPU dedicada)"),
    (55, "Prepárate: en 5 s, cambia a RENDIMIENTO -> GPU 0 y haz la captura 2"),
    (60, ">>> AHORA: captura 2 = RENDIMIENTO -> GPU 0 (memoria de GPU dedicada visible)"),
]


def _aviso(mensaje: str) -> None:
    print(f"\n  *** {mensaje} ***\n", file=sys.__stdout__, flush=True)
    try:  # pitido de Windows: se oye aunque el terminal quede tapado por otra ventana
        import winsound

        winsound.Beep(1000, 300)
        if mensaje.startswith(">>>"):
            winsound.Beep(1500, 300)  # el aviso de "AHORA" suena doble
    except Exception:
        pass  # sin sonido (otro sistema o sin altavoz): el mensaje en pantalla basta


def _marca() -> str:
    """Marca de tiempo con el mismo formato que la columna `timestamp` de nvidia-smi."""
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def _numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.astype(str).str.extract(r"([-\d.]+)")[0], errors="coerce")


def _resumen_nvidia_smi(ruta) -> None:
    """Máximos de utilización, memoria, potencia y temperatura del CSV de nvidia-smi."""
    try:
        t = pd.read_csv(ruta)
        t.columns = [c.strip() for c in t.columns]
        print(f"\nResumen de {ruta.name}: {len(t)} muestras")
        for columna, unidad in (("utilization.gpu [%]", "%"), ("utilization.memory [%]", "%"),
                                ("memory.used [MiB]", "MiB"), ("power.draw [W]", "W"),
                                ("temperature.gpu", "°C")):
            if columna in t.columns:
                v = _numero(t[columna])
                print(f"  {columna:24s} máximo {v.max():8.1f} {unidad}   promedio {v.mean():8.1f} {unidad}")
    except Exception as error:  # el resumen es solo informativo
        print(f"(No se pudo resumir el CSV: {error})")


def correr_evidencia(modelo_nombre: str, vueltas: int, espera: int, intervalo_ms: int,
                     avisos: bool = True) -> None:
    from src.calculo_gpu.modelos_gpu import crear_modelo
    from src.protocolo_parte2 import ejecutar_protocolo

    nombre = f"parte2_{modelo_nombre}_gpu"
    X, y = cargar_matrices()
    particiones = cargar_particiones()
    print(f"{modelo_nombre} / gpu: {len(X)} filas; {vueltas} vuelta(s) del protocolo completo")
    print("No se guardan tiempos ni métricas: es una pasada solo para nvidia-smi.")
    for s in range(espera, 0, -1):
        print(f"  empieza en {s} s ...", flush=True)
        time.sleep(1)

    fases = []
    temporizadores = []
    if avisos:
        for segundos, mensaje in AVISOS_CAPTURA:
            t = threading.Timer(segundos, _aviso, args=(mensaje,))
            t.daemon = True
            t.start()
            temporizadores.append(t)
    with registrar_gpu(nombre, intervalo_ms) as destino:
        time.sleep(1)  # unas muestras con la GPU en reposo
        inicio = _marca()
        modelo = crear_modelo(modelo_nombre, "gpu")  # calentamiento: 3 ajustes del tamaño real
        fases.append(("calentamiento", inicio, _marca()))
        print("  calentamiento terminado", flush=True)
        for v in range(1, vueltas + 1):
            inicio, t0 = _marca(), time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):  # sin la línea por experimento
                ejecutar_protocolo(modelo, X, y, particiones, calentar_antes=False)
            fases.append((f"vuelta {v}", inicio, _marca()))
            print(f"  vuelta {v}/{vueltas}: {time.perf_counter() - t0:.1f} s", flush=True)
        time.sleep(1)  # unas muestras con la GPU en reposo al final
    for t in temporizadores:  # si la corrida fue más corta, no avisar de más
        t.cancel()

    tabla = pd.DataFrame(fases, columns=["fase", "inicio", "fin"])
    ruta_fases = EVIDENCIAS / "nvidia-smi" / f"{nombre}_fases.csv"
    tabla.to_csv(ruta_fases, index=False)
    print(f"\nGuardado: {destino.relative_to(EVIDENCIAS.parent)}")
    print(f"Guardado: {ruta_fases.relative_to(EVIDENCIAS.parent)}")
    _resumen_nvidia_smi(destino)


def medir_copias(repeticiones: int, calentamiento: int, maquina: str) -> None:
    import cupy as cp

    X, y = cargar_matrices()
    e4 = cargar_particiones()["ej4_busqueda"]
    X_ent = np.ascontiguousarray(X[e4.entrenamiento])
    y_ent = np.ascontiguousarray(y[e4.entrenamiento])
    X_pru = np.ascontiguousarray(X[e4.prueba])
    proba64 = cp.zeros(len(X_pru), dtype=cp.float64)  # lo que devuelve la Regresión Logística
    proba32 = cp.zeros(len(X_pru), dtype=cp.float32)  # lo que devuelve XGBoost
    notas = f"ej4; maquina={maquina or '?'}"
    comunes = dict(version="GPU", repeticiones=repeticiones, calentamiento=calentamiento,
                   sincronizar_gpu=True, notas=notas)
    mb_ent = (X_ent.nbytes + y_ent.nbytes) / 1e6
    mediciones = [
        medir(lambda: (cp.asarray(X_ent), cp.asarray(y_ent)),
              fase=f"copias: X,y de entrenamiento CPU->GPU ({len(y_ent)} filas, {mb_ent:.1f} MB)", **comunes),
        medir(lambda: cp.asarray(X_pru),
              fase=f"copias: X de prueba CPU->GPU ({len(X_pru)} filas, {X_pru.nbytes / 1e6:.1f} MB)",
              **comunes),
        medir(lambda: cp.asnumpy(proba64),
              fase=f"copias: probabilidades GPU->CPU float64, Regresion Logistica ({len(X_pru)} valores)",
              **comunes),
        medir(lambda: cp.asnumpy(proba32),
              fase=f"copias: probabilidades GPU->CPU float32, XGBoost ({len(X_pru)} valores)", **comunes),
    ]
    guardar(mediciones, "parte2_copias_gpu.csv", sobrescribir=True)
    print(f"{'fase':100s} {'promedio_s':>10s} {'desv_s':>8s}")
    for m in mediciones:
        print(f"{m.fase:100s} {m.promedio_s:10.5f} {m.desviacion_s:8.5f}")
    print("\nGuardado: resultados/tiempos/parte2_copias_gpu.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidencia de nvidia-smi y copias CPU<->GPU (Parte II)")
    parser.add_argument("--modelo", choices=["regresion_logistica", "xgboost"],
                        help="modelo a registrar con nvidia-smi")
    parser.add_argument("--vueltas", type=int, default=None,
                        help="vueltas del protocolo (por defecto 15 en Regresión Logística, 1 en XGBoost)")
    parser.add_argument("--espera", type=int, default=10, help="segundos antes de empezar (grabación)")
    parser.add_argument("--intervalo-ms", type=int, default=200, help="muestreo de nvidia-smi")
    parser.add_argument("--sin-avisos", action="store_true",
                        help="no mostrar los avisos de captura a los 40 s y 60 s")
    parser.add_argument("--copias", action="store_true",
                        help="medir las copias CPU<->GPU en lugar de registrar nvidia-smi")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--calentamiento", type=int, default=3)
    parser.add_argument("--maquina", default="", help="etiqueta de la máquina (solo con --copias)")
    args = parser.parse_args()

    if args.copias:
        medir_copias(args.repeticiones, args.calentamiento, args.maquina)
        return
    if not args.modelo:
        parser.error("indica --modelo regresion_logistica|xgboost, o --copias")
    vueltas = args.vueltas or VUELTAS_POR_DEFECTO[args.modelo]
    correr_evidencia(args.modelo, vueltas, args.espera, args.intervalo_ms,
                     avisos=not args.sin_avisos)


if __name__ == "__main__":
    main()
