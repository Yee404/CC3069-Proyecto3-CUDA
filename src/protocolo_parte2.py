"""Orquestador único del protocolo completo de la Parte II (PLAN.md §6.2). María José Yee.

Reproduce los 9 experimentos del notebook 04 con las particiones fijas de
`preparacion_cpu.dataset` (75 ajustes de Regresión Logística y 93 de XGBoost) y las métricas
de `evaluacion.py`. NO conoce la implementación del modelo: recibe un objeto con este contrato
(el mismo que cumplen `referencia_cpu/modelos_cpu.py` y la futura versión GPU):

    nombre, variante, version, rejilla, ajustes_en_paralelo, sincronizar_gpu
    ajustar(X, y, params)    -> modelo ajustado (X float32 C-contiguo, y int8)
    probabilidad(modelo, X)  -> ndarray NumPy (n,) en el host con P(y = 1)
                                (en GPU, la copia al host se hace dentro y se cronometra)

Este módulo vive fuera de `calculo_gpu/` porque lee y escribe archivos.

Qué hace cada tipo de experimento:
    holdout  búsqueda con la CV interna (G combinaciones x 3 pliegues), reajuste con todo el
             entrenamiento y evaluación en `prueba`  -> 3G + 1 ajustes
    cv       5 pliegues con los mejores hiperparámetros de `ej4_busqueda` (sin búsqueda)
             -> 5 ajustes. Por eso `ej4_busqueda` siempre corre primero.

Tiempos (PLAN.md §7):
    * Una pasada completa por versión: tiempo de pared de cada experimento (ajuste +
      predicción; las métricas no cuentan porque son iguales en todas las versiones).
    * Repeticiones con `medicion.tiempos.medir` (calentamiento + 5) sobre tres unidades
      representativas: la búsqueda de ej4, un entrenamiento final y una predicción.

Uso (desde la raíz del repositorio):

    .venv\\Scripts\\python -m src.protocolo_parte2 --modelo xgboost --variante cpu_secuencial
    .venv\\Scripts\\python -m src.protocolo_parte2 --modelo regresion_logistica \\
        --variante cpu_multinucleo --experimentos ej7_intra_amatitlan --sin-representativos

Salidas:
    resultados/validacion/parte2_<modelo>_<variante>_<experimento>.csv            métricas
    resultados/validacion/parte2_<modelo>_<variante>_<experimento>_busqueda.csv   (holdout)
    resultados/validacion/parte2_<modelo>_<variante>.json                         resumen
    resultados/tiempos/parte2_<modelo>_<variante>.csv                             tiempos
Si se corre un subconjunto de experimentos, los nombres llevan el sufijo `_parcial`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from importlib import metadata
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from sklearn.metrics import average_precision_score

from src import RESULTADOS
from src.evaluacion import metricas, umbral_optimo
from src.medicion.tiempos import Medicion, guardar, medir
from src.parametros_modelos import combinaciones
from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones

ORDEN_EXPERIMENTOS = [
    "ej4_busqueda", "ej6_aleatoria", "ej6_espacial", "ej6_temporal", "ej6_cronologico",
    "ej7_atitlan_a_amatitlan", "ej7_amatitlan_a_atitlan", "ej7_intra_atitlan",
    "ej7_intra_amatitlan",
]
AJUSTES_ESPERADOS = {"regresion_logistica": 75, "xgboost": 93}


# --------------------------------------------------------------------------
# Ejecución de ajustes
# --------------------------------------------------------------------------
def _sincronizar(modelo) -> None:
    if modelo.sincronizar_gpu:
        from numba import cuda

        cuda.synchronize()


def ajustar_y_predecir(modelo, X, y, indices_ent, indices_eval, params):
    """Un ajuste + su predicción. Devuelve (probabilidad float64, s_ajuste, s_prediccion)."""
    X_ent, y_ent, X_eval = X[indices_ent], y[indices_ent], X[indices_eval]
    inicio = perf_counter()
    ajustado = modelo.ajustar(X_ent, y_ent, params)
    _sincronizar(modelo)
    medio = perf_counter()
    proba = modelo.probabilidad(ajustado, X_eval)
    _sincronizar(modelo)
    fin = perf_counter()
    # float64 siempre, para que CPU y GPU entren igual a las métricas.
    return np.asarray(proba, dtype=np.float64), medio - inicio, fin - medio


def ejecutar_lote(modelo, X, y, tareas, n_procesos):
    """Corre una lista de (ent, eval, params). Devuelve (salidas, segundos de pared).

    Si el modelo declara `ajustes_en_paralelo`, los ajustes se reparten entre procesos con
    joblib (tamaño de pool constante para que loky reutilice los procesos).
    """
    inicio = perf_counter()
    if modelo.ajustes_en_paralelo and len(tareas) > 1:
        salidas = Parallel(n_jobs=n_procesos)(
            delayed(ajustar_y_predecir)(modelo, X, y, ent, ev, p) for ent, ev, p in tareas
        )
    else:
        salidas = [ajustar_y_predecir(modelo, X, y, ent, ev, p) for ent, ev, p in tareas]
    return salidas, perf_counter() - inicio


def calentar(modelo, X, y, n_procesos) -> None:
    """Ajuste pequeño descartado: compila JIT, arranca los procesos y sube la GPU."""
    idx = np.concatenate([np.flatnonzero(y == 1)[:300], np.flatnonzero(y == 0)[:5000]])
    idx = idx.astype(np.int32)
    params = combinaciones(modelo.rejilla)[0]
    n = n_procesos if modelo.ajustes_en_paralelo else 1
    ejecutar_lote(modelo, X, y, [(idx, idx, params)] * max(n, 2 if n > 1 else 1), n_procesos)


# --------------------------------------------------------------------------
# Experimentos
# --------------------------------------------------------------------------
def _extra(modelo, exp, etapa, criterio, mejores, n_ent, y_eval, **otros) -> dict:
    return {
        "algoritmo": modelo.nombre, "variante": modelo.variante, "experimento": exp.nombre,
        "etapa": etapa, "criterio_umbral": criterio,
        "hiperparametros": json.dumps(mejores, ensure_ascii=False),
        "n_entrenamiento": int(n_ent), "n_evaluacion": int(len(y_eval)),
        "positivos_evaluacion": int(np.sum(y_eval == 1)), **otros,
    }


def _sumas(salidas) -> tuple[float, float]:
    return sum(s[1] for s in salidas), sum(s[2] for s in salidas)


def buscar_hiperparametros(modelo, X, y, exp, n_procesos):
    """Equivalente a GridSearchCV(scoring="average_precision", refit=False)."""
    combos = combinaciones(modelo.rejilla)
    tareas = [(ent, val, p) for p in combos for ent, val in exp.pliegues]
    salidas, segundos = ejecutar_lote(modelo, X, y, tareas, n_procesos)
    filas, k = [], 0
    for p in combos:
        puntajes = []
        for _, val in exp.pliegues:
            puntajes.append(float(average_precision_score(y[val], salidas[k][0])))
            k += 1
        filas.append({
            **p, **{f"pr_auc_pliegue{i}": v for i, v in enumerate(puntajes)},
            "pr_auc_media": float(np.mean(puntajes)), "pr_auc_desviacion": float(np.std(puntajes)),
        })
    mejor = max(filas, key=lambda f: f["pr_auc_media"])  # primera de mayor puntaje
    mejores = {c: mejor[c] for c in modelo.rejilla}
    return mejores, filas, len(tareas), segundos, _sumas(salidas)


def correr_holdout(modelo, X, y, exp, n_procesos):
    mejores, busqueda, n_busq, t_busq, (s_aj, s_pr) = buscar_hiperparametros(
        modelo, X, y, exp, n_procesos)
    salidas, t_final = ejecutar_lote(
        modelo, X, y, [(exp.entrenamiento, exp.prueba, mejores)], n_procesos)
    proba, y_pru = salidas[0][0], y[exp.prueba]
    umbral, _ = umbral_optimo(y_pru, proba)
    filas = [
        metricas(modelo.nombre, y_pru, proba, u,
                 _extra(modelo, exp, "prueba", criterio, mejores, len(exp.entrenamiento), y_pru))
        for criterio, u in (("0.5", 0.5), ("optimo F2", umbral))
    ]
    tiempos = {"n_ajustes": n_busq + 1, "tiempo_pared_s": t_busq + t_final,
               "suma_ajustes_s": s_aj + salidas[0][1], "suma_predicciones_s": s_pr + salidas[0][2]}
    return mejores, filas, busqueda, tiempos


def correr_cv(modelo, X, y, exp, mejores, n_procesos):
    tareas = [(ent, val, mejores) for ent, val in exp.pliegues]
    salidas, segundos = ejecutar_lote(modelo, X, y, tareas, n_procesos)
    filas, ys, ps = [], [], []
    for k, ((ent, val), (proba, _, _)) in enumerate(zip(exp.pliegues, salidas)):
        filas.append(metricas(modelo.nombre, y[val], proba, 0.5, _extra(
            modelo, exp, "pliegue", "0.5", mejores, len(ent), y[val], pliegue=k)))
        ys.append(y[val])
        ps.append(proba)
    y_fuera, p_fuera = np.concatenate(ys), np.concatenate(ps)
    umbral, _ = umbral_optimo(y_fuera, p_fuera)
    for criterio, u in (("0.5", 0.5), ("optimo F2", umbral)):
        filas.append(metricas(modelo.nombre, y_fuera, p_fuera, u, _extra(
            modelo, exp, "fuera_de_pliegue", criterio, mejores,
            np.mean([len(e) for e, _ in exp.pliegues]), y_fuera)))
    s_aj, s_pr = _sumas(salidas)
    tiempos = {"n_ajustes": len(tareas), "tiempo_pared_s": segundos,
               "suma_ajustes_s": s_aj, "suma_predicciones_s": s_pr}
    return filas, tiempos


def ejecutar_protocolo(modelo, X, y, particiones, experimentos=None, n_procesos=None,
                       calentar_antes=True) -> dict:
    """Corre el protocolo con CUALQUIER modelo que cumpla el contrato del módulo."""
    n_procesos = n_procesos or cpu_count(only_physical_cores=True)
    desconocidos = set(particiones) ^ set(ORDEN_EXPERIMENTOS)
    if desconocidos:
        raise ValueError(f"Las particiones no coinciden con el protocolo: {sorted(desconocidos)}")
    pedidos = set(experimentos or ORDEN_EXPERIMENTOS)
    if set(pedidos) - set(ORDEN_EXPERIMENTOS):
        raise ValueError(f"Experimentos desconocidos: {sorted(pedidos - set(ORDEN_EXPERIMENTOS))}")
    if any(particiones[e].tipo == "cv" for e in pedidos) and "ej4_busqueda" not in pedidos:
        print("Aviso: los experimentos 'cv' usan los mejores hiperparámetros de ej4_busqueda; "
              "se agrega ej4_busqueda.")
        pedidos.add("ej4_busqueda")
    seleccion = [e for e in ORDEN_EXPERIMENTOS if e in pedidos]

    if calentar_antes:
        print("Calentamiento (descartado)...")
        calentar(modelo, X, y, n_procesos)

    resultado = {"metricas": {}, "busquedas": {}, "mejores": {}, "tiempos": {},
                 "n_ajustes": 0, "seleccion": seleccion, "n_procesos": n_procesos}
    mejores_ej4 = None
    for nombre in seleccion:
        exp = particiones[nombre]
        if exp.tipo == "holdout":
            mejores, filas, busqueda, tiempos = correr_holdout(modelo, X, y, exp, n_procesos)
            resultado["busquedas"][nombre] = busqueda
            resultado["mejores"][nombre] = mejores
            esperado = len(exp.pliegues) * len(combinaciones(modelo.rejilla)) + 1
            if nombre == "ej4_busqueda":
                mejores_ej4 = mejores
        else:
            filas, tiempos = correr_cv(modelo, X, y, exp, mejores_ej4, n_procesos)
            esperado = len(exp.pliegues)
        if tiempos["n_ajustes"] != esperado:
            raise RuntimeError(f"{nombre}: {tiempos['n_ajustes']} ajustes, se esperaban {esperado}")
        resultado["metricas"][nombre] = filas
        resultado["tiempos"][nombre] = tiempos
        resultado["n_ajustes"] += tiempos["n_ajustes"]
        print(f"  {nombre:26s} {tiempos['n_ajustes']:3d} ajustes  "
              f"{tiempos['tiempo_pared_s']:8.2f} s")

    if seleccion == ORDEN_EXPERIMENTOS:
        esperado_total = AJUSTES_ESPERADOS.get(modelo.nombre)
        if esperado_total is not None and resultado["n_ajustes"] != esperado_total:
            raise RuntimeError(f"Total de ajustes {resultado['n_ajustes']} != {esperado_total}")
    return resultado


# --------------------------------------------------------------------------
# Tiempos representativos (calentamiento + repeticiones con medir)
# --------------------------------------------------------------------------
def medir_representativos(modelo, X, y, particiones, mejores_ej4, n_procesos,
                          repeticiones, calentamiento, notas="") -> list[Medicion]:
    e4 = particiones["ej4_busqueda"]
    combos = combinaciones(modelo.rejilla)
    tareas = [(ent, val, p) for p in combos for ent, val in e4.pliegues]
    X_ent, y_ent, X_pru = X[e4.entrenamiento], y[e4.entrenamiento], X[e4.prueba]
    comunes = dict(version=modelo.version, repeticiones=repeticiones,
                   calentamiento=calentamiento, sincronizar_gpu=modelo.sincronizar_gpu)

    mediciones = [medir(
        lambda: ejecutar_lote(modelo, X, y, tareas, n_procesos),
        fase=f"representativo: busqueda ej4 ({len(tareas)} ajustes + prediccion de validacion)",
        notas=f"{modelo.variante}; {n_procesos} procesos; {notas}", **comunes)]
    mediciones.append(medir(
        lambda: modelo.ajustar(X_ent, y_ent, mejores_ej4),
        fase=f"representativo: entrenamiento ej4 final ({len(y_ent)} filas)",
        notas=f"{modelo.variante}; {notas}", **comunes))
    final = modelo.ajustar(X_ent, y_ent, mejores_ej4)
    mediciones.append(medir(
        lambda: modelo.probabilidad(final, X_pru),
        fase=f"representativo: prediccion ej4 prueba ({len(X_pru)} filas)",
        notas=f"{modelo.variante}; {notas}", **comunes))
    return mediciones


# --------------------------------------------------------------------------
# Guardado
# --------------------------------------------------------------------------
def _version_paquete(nombre: str) -> str:
    try:
        return metadata.version(nombre)
    except metadata.PackageNotFoundError:
        return "?"


def _a_python(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))


def guardar_resultados(modelo, resultado, parametros_corrida: dict) -> str:
    parcial = resultado["seleccion"] != ORDEN_EXPERIMENTOS
    base = f"parte2_{modelo.nombre}_{modelo.variante}" + ("_parcial" if parcial else "")
    carpeta = RESULTADOS / "validacion"
    carpeta.mkdir(parents=True, exist_ok=True)
    for exp, filas in resultado["metricas"].items():
        pd.DataFrame(filas).to_csv(carpeta / f"{base}_{exp}.csv", index=False)
    for exp, filas in resultado["busquedas"].items():
        pd.DataFrame(filas).to_csv(carpeta / f"{base}_{exp}_busqueda.csv", index=False)
    resumen = {
        "algoritmo": modelo.nombre, "variante": modelo.variante, "version": modelo.version,
        "parcial": parcial, "n_ajustes": resultado["n_ajustes"],
        "n_ajustes_esperados": AJUSTES_ESPERADOS.get(modelo.nombre),
        "experimentos": resultado["seleccion"],
        "mejores_hiperparametros": resultado["mejores"],
        "tiempos_por_experimento": resultado["tiempos"],
        "tiempo_pared_total_s": sum(t["tiempo_pared_s"] for t in resultado["tiempos"].values()),
        "n_procesos": resultado["n_procesos"],
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "paquetes": {p: _version_paquete(p) for p in ("scikit-learn", "xgboost", "numpy")},
        **parametros_corrida,
    }
    (carpeta / f"{base}.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=_a_python), encoding="utf-8")
    return base


def mediciones_de_la_pasada(modelo, resultado, notas="") -> list[Medicion]:
    """Una Medicion por experimento y una total (1 pasada, sin desviación)."""
    mediciones = []
    for exp, t in resultado["tiempos"].items():
        mediciones.append(Medicion(
            fase=f"protocolo {exp}", version=modelo.version, promedio_s=t["tiempo_pared_s"],
            desviacion_s=0.0, corridas_s=[t["tiempo_pared_s"]],
            notas=f"1 pasada; {t['n_ajustes']} ajustes; sin metricas; {notas}"))
    total = sum(t["tiempo_pared_s"] for t in resultado["tiempos"].values())
    mediciones.append(Medicion(
        fase="protocolo completo" if resultado["seleccion"] == ORDEN_EXPERIMENTOS
        else "protocolo (subconjunto)", version=modelo.version, promedio_s=total,
        desviacion_s=0.0, corridas_s=[total],
        notas=f"1 pasada; {resultado['n_ajustes']} ajustes; {notas}"))
    return mediciones


# --------------------------------------------------------------------------
# Línea de comandos
# --------------------------------------------------------------------------
def _crear_modelo(nombre: str, variante: str):
    if variante.startswith("cpu"):
        from src.referencia_cpu.modelos_cpu import crear_modelo
    else:
        try:
            from src.calculo_gpu.modelos_gpu import crear_modelo
        except ImportError as error:
            raise SystemExit(
                "La versión GPU aún no existe: falta src/calculo_gpu/modelos_gpu.py con "
                f"crear_modelo(nombre, 'gpu') ({error})")
    return crear_modelo(nombre, variante)


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocolo completo de la Parte II (PLAN §6.2)")
    parser.add_argument("--modelo", required=True, choices=["regresion_logistica", "xgboost"])
    parser.add_argument("--variante", required=True,
                        choices=["cpu_secuencial", "cpu_multinucleo", "gpu"])
    parser.add_argument("--experimentos", nargs="+", default=None, choices=ORDEN_EXPERIMENTOS,
                        help="subconjunto (por defecto, los 9)")
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--calentamiento", type=int, default=1)
    parser.add_argument("--procesos", type=int, default=None,
                        help="procesos para repartir ajustes (por defecto, núcleos físicos)")
    parser.add_argument("--sin-representativos", action="store_true",
                        help="omitir las repeticiones con medir()")
    parser.add_argument("--maquina", default="", help="etiqueta de la máquina (resultados/maquinas)")
    args = parser.parse_args()

    modelo = _crear_modelo(args.modelo, args.variante)
    X, y = cargar_matrices()
    particiones = cargar_particiones()
    print(f"{modelo.nombre} / {modelo.variante}: {len(X)} filas, {int(np.sum(y == 1))} positivos")

    resultado = ejecutar_protocolo(modelo, X, y, particiones, args.experimentos, args.procesos)
    notas = f"maquina={args.maquina}" if args.maquina else ""
    mediciones = mediciones_de_la_pasada(modelo, resultado, notas)

    if not args.sin_representativos:
        print("\nTiempos representativos (calentamiento + repeticiones)...")
        mediciones += medir_representativos(
            modelo, X, y, particiones, resultado["mejores"]["ej4_busqueda"],
            resultado["n_procesos"], args.repeticiones, args.calentamiento, notas)

    base = guardar_resultados(modelo, resultado, {
        "repeticiones": args.repeticiones, "calentamiento": args.calentamiento,
        "maquina": args.maquina, "representativos": not args.sin_representativos})
    tabla = guardar(mediciones, f"{base}.csv", sobrescribir=True)
    print("\n" + tabla[["fase", "version", "promedio_s", "desviacion_s"]].round(3).to_string())
    print(f"\nGuardado: resultados/validacion/{base}*.csv|json y resultados/tiempos/{base}.csv")


if __name__ == "__main__":
    # Se importa con su nombre real para que joblib serialice las funciones por referencia.
    from src.protocolo_parte2 import main as _main

    _main()
