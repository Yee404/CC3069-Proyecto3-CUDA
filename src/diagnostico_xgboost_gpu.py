"""Diagnóstico de dónde se va el tiempo de XGBoost en GPU vs CPU. María José Yee.

Mide UN ajuste representativo del protocolo (la reentrenada final de `ej4_busqueda`:
281 159 filas de entrenamiento, learning_rate=0.1, max_leaves=31, 100 árboles) y su
predicción sobre la partición de prueba (120 497 filas), separando cada fase:

    GPU: copia X,y a la GPU | QuantileDMatrix | entrenamiento | copia X de prueba a la GPU |
         predicción | copia del resultado al host
    CPU: QuantileDMatrix | entrenamiento | predicción        (1 hilo y todos los hilos)

Además mide la ruta real que usa el protocolo (`modelos_gpu` / `modelos_cpu`: `ajustar` y
`probabilidad`) para comprobar que el desglose no esconde ningún costo.

Usa `src/medicion/tiempos.py`: 3 calentamientos + 5 repeticiones por defecto, promedio y
desviación. Los hiperparámetros salen de `src/parametros_modelos.py` (no se tocan).

Este script vive fuera de `calculo_gpu/` porque lee y escribe archivos. NO modifica
`protocolo_parte2.py`.

Uso (desde la raíz del repositorio):

    .venv\\Scripts\\python -m src.diagnostico_xgboost_gpu
    .venv\\Scripts\\python -m src.diagnostico_xgboost_gpu --filas 3750000   # escalabilidad

Con `--filas N` se REPLICA el entrenamiento hasta N filas (y la prueba en la misma
proporción) SOLO para medir tiempos: las filas repetidas no producen un modelo válido y
las métricas no se miran. Es un experimento de escalabilidad, no cambia el protocolo.

Salida: resultados/tiempos/parte2_diagnostico_xgboost.csv
        (o ..._escala_<N>.csv con --filas).
"""

from __future__ import annotations

import argparse
import os
import statistics
from time import perf_counter

import cupy as cp
import numpy as np
import xgboost as xgb

from src.calculo_gpu.modelos_gpu import crear_modelo as crear_modelo_gpu
from src.medicion.tiempos import Medicion, guardar, medir
from src.parametros_modelos import parametros_xgboost, peso_positivos
from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones
from src.referencia_cpu.modelos_cpu import crear_modelo as crear_modelo_cpu

HIPERPARAMETROS = {"learning_rate": 0.1, "max_leaves": 31}  # ajuste representativo
NUCLEOS_LOGICOS = os.cpu_count() or 1

FASES_GPU = ("copia X,y a GPU", "QuantileDMatrix", "entrenamiento",
             "copia X prueba a GPU", "prediccion", "copia resultado a CPU")
FASES_CPU = ("QuantileDMatrix", "entrenamiento", "prediccion")


# --------------------------------------------------------------------------
# Parámetros del núcleo de XGBoost (los mismos que usa XGBClassifier)
# --------------------------------------------------------------------------
def _parametros_nucleo(device: str, hilos: int, y_ent) -> tuple[dict, int]:
    """Devuelve (params para xgb.train, número de árboles) idénticos a los del protocolo."""
    clasificador = xgb.XGBClassifier(**parametros_xgboost(
        **HIPERPARAMETROS, scale_pos_weight=peso_positivos(y_ent), device=device, n_jobs=hilos))
    params = {k: v for k, v in clasificador.get_xgb_params().items() if k != "n_jobs"}
    params["nthread"] = hilos
    return params, int(clasificador.get_params()["n_estimators"])


def _sincronizar() -> None:
    cp.cuda.Device().synchronize()


def _marca(registro: dict, nombre: str, inicio: float, gpu: bool) -> float:
    """Cierra una fase (sincronizando la GPU si hace falta) y devuelve el inicio de la siguiente."""
    if gpu:
        _sincronizar()
    ahora = perf_counter()
    registro[nombre] = ahora - inicio
    return ahora


# --------------------------------------------------------------------------
# Una pasada desglosada
# --------------------------------------------------------------------------
def pasada_gpu(X_ent, y_ent, X_pru) -> tuple[dict, np.ndarray]:
    params, rondas = _parametros_nucleo("cuda", 1, y_ent)
    reg: dict = {}
    t = perf_counter()
    X_g, y_g = cp.asarray(X_ent), cp.asarray(y_ent)
    t = _marca(reg, "copia X,y a GPU", t, True)
    dtrain = xgb.QuantileDMatrix(X_g, label=y_g, max_bin=params["max_bin"])
    t = _marca(reg, "QuantileDMatrix", t, True)
    booster = xgb.train(params, dtrain, num_boost_round=rondas)
    t = _marca(reg, "entrenamiento", t, True)
    X_pru_g = cp.asarray(X_pru)
    t = _marca(reg, "copia X prueba a GPU", t, True)
    proba_g = booster.inplace_predict(X_pru_g)
    t = _marca(reg, "prediccion", t, True)
    proba = cp.asnumpy(proba_g)
    _marca(reg, "copia resultado a CPU", t, True)
    return reg, proba


def pasada_cpu(X_ent, y_ent, X_pru, hilos: int) -> tuple[dict, np.ndarray]:
    params, rondas = _parametros_nucleo("cpu", hilos, y_ent)
    reg: dict = {}
    t = perf_counter()
    dtrain = xgb.QuantileDMatrix(X_ent, label=y_ent, max_bin=params["max_bin"], nthread=hilos)
    t = _marca(reg, "QuantileDMatrix", t, False)
    booster = xgb.train(params, dtrain, num_boost_round=rondas)
    t = _marca(reg, "entrenamiento", t, False)
    proba = booster.inplace_predict(X_pru)
    _marca(reg, "prediccion", t, False)
    return reg, proba


# --------------------------------------------------------------------------
# Medición con el protocolo de tiempos.py
# --------------------------------------------------------------------------
def medir_desglose(pasada, fases, version, gpu, repeticiones, calentamiento, notas) -> list[Medicion]:
    """Repite la pasada con `medir` y agrega promedio y desviación por fase."""
    registros: list[dict] = []

    def una() -> None:
        registros.append(pasada()[0])

    total = medir(una, fase="diagnostico: pasada completa (suma de fases)", version=version,
                  repeticiones=repeticiones, calentamiento=calentamiento,
                  sincronizar_gpu=gpu, notas=notas)
    usados = registros[calentamiento:]
    assert len(usados) == repeticiones, (len(usados), repeticiones)
    salida = []
    for fase in fases:
        valores = [r[fase] for r in usados]
        salida.append(Medicion(
            fase=f"diagnostico: {fase}", version=version, promedio_s=statistics.mean(valores),
            desviacion_s=statistics.stdev(valores) if len(valores) > 1 else 0.0,
            corridas_s=valores, notas=notas))
    return salida + [total]


def medir_ruta_real(modelo, X_ent, y_ent, X_pru, repeticiones, calentamiento, notas) -> list[Medicion]:
    """Ruta del protocolo: ajustar() y probabilidad() tal cual (incluye copias en GPU)."""
    comunes = dict(version=modelo.version, repeticiones=repeticiones, calentamiento=calentamiento,
                   sincronizar_gpu=modelo.sincronizar_gpu, notas=notas)
    ajuste = medir(lambda: modelo.ajustar(X_ent, y_ent, HIPERPARAMETROS),
                   fase="diagnostico: ruta real ajustar()", **comunes)
    ajustado = modelo.ajustar(X_ent, y_ent, HIPERPARAMETROS)
    prediccion = medir(lambda: modelo.probabilidad(ajustado, X_pru),
                       fase="diagnostico: ruta real probabilidad()", **comunes)
    return [ajuste, prediccion]


def _replicar(a: np.ndarray, n: int) -> np.ndarray:
    """Repite las filas de `a` hasta tener n filas (solo para pruebas de escalabilidad)."""
    veces = -(-n // len(a))
    return np.ascontiguousarray(np.tile(a, (veces,) + (1,) * (a.ndim - 1))[:n])


def _imprimir(mediciones: list[Medicion]) -> None:
    print(f"\n{'fase':58s} {'version':16s} {'promedio_s':>10s} {'desv_s':>8s}")
    for m in mediciones:
        print(f"{m.fase:58s} {m.version:16s} {m.promedio_s:10.4f} {m.desviacion_s:8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de tiempos de XGBoost GPU vs CPU")
    parser.add_argument("--filas", type=int, default=None,
                        help="replicar el entrenamiento hasta N filas (escalabilidad)")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--calentamiento", type=int, default=3)
    parser.add_argument("--versiones", nargs="+", default=["gpu", "cpu_secuencial", "cpu_multinucleo"],
                        choices=["gpu", "cpu_secuencial", "cpu_multinucleo"])
    parser.add_argument("--maquina", default="", help="etiqueta de la máquina")
    args = parser.parse_args()

    X, y = cargar_matrices()
    e4 = cargar_particiones()["ej4_busqueda"]
    X_ent, y_ent, X_pru = X[e4.entrenamiento], y[e4.entrenamiento], X[e4.prueba]
    if args.filas:
        n_pru = round(args.filas * len(X_pru) / len(X_ent))
        X_ent, y_ent, X_pru = _replicar(X_ent, args.filas), _replicar(y_ent, args.filas), _replicar(X_pru, n_pru)
    datos = "replicado (solo tiempos)" if args.filas else "real"
    notas = (f"datos {datos}; {len(X_ent)} filas de entrenamiento, {len(X_pru)} de prueba; "
             f"lr 0.1, 31 hojas, 100 arboles; maquina={args.maquina or '?'}")
    print(notas)

    mediciones: list[Medicion] = []
    if "gpu" in args.versiones:
        # Comprobación única: el desglose (xgb.train) y la ruta real dan las mismas probabilidades.
        modelo_gpu = crear_modelo_gpu("xgboost", "gpu")
        p_real = modelo_gpu.probabilidad(modelo_gpu.ajustar(X_ent, y_ent, HIPERPARAMETROS), X_pru)
        diferencia = float(np.max(np.abs(p_real - pasada_gpu(X_ent, y_ent, X_pru)[1])))
        print(f"GPU: diferencia maxima entre desglose y ruta real = {diferencia:.2e}")
        nota_gpu = notas + f"; dif. max. desglose vs ruta real {diferencia:.1e}"
        print("Midiendo GPU...")
        mediciones += medir_desglose(lambda: pasada_gpu(X_ent, y_ent, X_pru), FASES_GPU, "GPU", True,
                                     args.repeticiones, args.calentamiento, nota_gpu)
        mediciones += medir_ruta_real(modelo_gpu, X_ent, y_ent, X_pru,
                                      args.repeticiones, args.calentamiento, nota_gpu)
    for variante, hilos in (("cpu_secuencial", 1), ("cpu_multinucleo", NUCLEOS_LOGICOS)):
        if variante not in args.versiones:
            continue
        modelo = crear_modelo_cpu("xgboost", variante)
        nota_cpu = notas + f"; {hilos} hilo(s)"
        print(f"Midiendo {variante} ({hilos} hilos)...")
        mediciones += medir_desglose(lambda h=hilos: pasada_cpu(X_ent, y_ent, X_pru, h), FASES_CPU,
                                     modelo.version, False, args.repeticiones, args.calentamiento, nota_cpu)
        mediciones += medir_ruta_real(modelo, X_ent, y_ent, X_pru,
                                      args.repeticiones, args.calentamiento, nota_cpu)

    _imprimir(mediciones)
    nombre = "parte2_diagnostico_xgboost" + (f"_escala_{args.filas}" if args.filas else "") + ".csv"
    guardar(mediciones, nombre, sobrescribir=True)
    print(f"\nGuardado: resultados/tiempos/{nombre}")


if __name__ == "__main__":
    from src.diagnostico_xgboost_gpu import main as _main

    _main()
