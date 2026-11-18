"""Validación de XGBoost GPU contra XGBoost CPU (T5). María José Yee.

Compara las métricas del protocolo completo (93 ajustes) de las dos versiones y reporta las
diferencias; NO fuerza igualdad (las sumas en GPU se hacen en otro orden, CONTEXTO_INFORME §2.4).

Dos partes (desde la raíz del repositorio):

    .venv\\Scripts\\python -m src.validar_xgboost_gpu
    .venv\\Scripts\\python -m src.validar_xgboost_gpu --cpu cpu_multinucleo

1) Los 9 experimentos, leyendo los CSV que dejó `protocolo_parte2.py` (no reentrena nada):
       resultados/validacion/parte2_xgboost_<variante>_<experimento>.csv
   Criterio B1 (el mismo de la Regresión Logística, §11.3), fijado el 2026-11-17 ANTES de ver
   las métricas de XGBoost GPU:
       B1  |PR-AUC GPU - PR-AUC CPU| <= 2e-3     (decide, uno por experimento)
   Se reporta, sin decidir: si los dos dispositivos eligieron los mismos hiperparámetros, y la
   diferencia de ROC-AUC, F2 y precision/recall con umbral 0.5.
   En XGBoost no hay un "óptimo de referencia" como con tol=1e-10 en la Regresión Logística,
   así que A, B2 y B3 no aplican.

2) Prueba con hiperparámetros IDÉNTICOS: reajusta en CPU y GPU las 4 combinaciones de la
   rejilla con el entrenamiento de ej4_busqueda y compara las probabilidades sobre la prueba
   (diferencia máxima y media, correlación, filas que cambian de clase con umbral 0.5, y
   diferencia de PR-AUC y ROC-AUC). Separa "el modelo difiere" de "se eligieron otros
   hiperparámetros". Solo se reporta.

3) Prueba extendida (`--extendida`): para los experimentos que NO pasan B1 (se leen de
   `xgboost_gpu_vs_cpu.csv`, o se eligen con `--experimentos`), con los hiperparámetros que
   eligió la GPU en el protocolo y los MISMOS datos de entrenamiento de cada experimento:
       * GPU vs CPU: ΔPR-AUC, ΔROC-AUC y filas que cambian de clase (umbral 0.5).
       * Línea base de ruido: la CPU contra sí misma entrenada sin una submuestra mínima
         (0.1 % de las filas de entrenamiento, 3 semillas fijas), mismas métricas. Mide cuánto
         se mueven los resultados SIN cambiar el hardware.
       * Repetibilidad: dos ajustes de GPU seguidos con los mismos datos.
   Regla de lectura, fijada el 2026-11-17 antes de correrla: la desviación de la GPU es
   "comparable al ruido" si |ΔPR-AUC GPU-CPU| <= 2 x (máximo |ΔPR-AUC| de la línea base);
   si es mayor, se marca "mayor que el ruido: investigar" (por ejemplo lossguide con
   max_depth=0 en GPU). Es solo una lectura: B1 no cambia y el FALLO se reporta tal cual.
   En los experimentos "cv" (ej6_temporal) se ajusta cada pliegue y las métricas son las
   de las predicciones fuera de pliegue juntas.
   Salida: resultados/validacion/xgboost_gpu_prueba_extendida.csv

4) Repetibilidad entre corridas del protocolo (`--comparar-con CARPETA`): compara los CSV
   `parte2_xgboost_gpu_*.csv` de `resultados/validacion` con los de otra carpeta (por ejemplo
   el respaldo de una corrida anterior). Salida: xgboost_gpu_repetibilidad_corridas.csv

Salidas:
    resultados/validacion/xgboost_gpu_vs_cpu.csv
    resultados/validacion/xgboost_gpu_prueba_identica.csv
Este script lee archivos, por eso vive fuera de `calculo_gpu/`. No mide tiempos.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import RESULTADOS
from src.parametros_modelos import combinaciones, REJILLA_XGBOOST
from src.protocolo_parte2 import ORDEN_EXPERIMENTOS

UMBRALES = {"B1_dif_pr_auc": 2e-3}  # |PR-AUC GPU - PR-AUC CPU|
NOTA_UMBRALES = (
    "Umbral B1 fijado el 2026-11-17 antes de ver las métricas de XGBoost GPU; es el mismo "
    "de la Regresión Logística (CONTEXTO_INFORME §11.3). A, B2 y B3 no aplican a XGBoost."
)
CARPETA = RESULTADOS / "validacion"


def _leer(variante: str, experimento: str) -> pd.DataFrame:
    ruta = CARPETA / f"parte2_xgboost_{variante}_{experimento}.csv"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta.relative_to(RESULTADOS.parent)}: corre el protocolo "
                         f"completo de XGBoost en la variante {variante}.")
    return pd.read_csv(ruta)


def _fila_principal(tabla: pd.DataFrame, criterio: str) -> pd.Series:
    """La fila de prueba (holdout) o fuera de pliegue (cv) con el criterio de umbral dado."""
    sel = tabla[tabla["etapa"].isin(["prueba", "fuera_de_pliegue"])
                & (tabla["criterio_umbral"].astype(str) == criterio)]
    if len(sel) != 1:
        raise SystemExit(f"Se esperaba 1 fila con criterio {criterio!r}, hay {len(sel)}")
    return sel.iloc[0]


def comparar_experimentos(variante_cpu: str) -> pd.DataFrame:
    filas = []
    for exp in ORDEN_EXPERIMENTOS:
        c, g = _leer(variante_cpu, exp), _leer("gpu", exp)
        c5, g5 = _fila_principal(c, "0.5"), _fila_principal(g, "0.5")
        co, go = _fila_principal(c, "optimo F2"), _fila_principal(g, "optimo F2")
        h_c, h_g = json.loads(c5["hiperparametros"]), json.loads(g5["hiperparametros"])
        filas.append({
            "experimento": exp, "etapa": c5["etapa"],
            "hiper_cpu": json.dumps(h_c), "hiper_gpu": json.dumps(h_g),
            "mismos_hiperparametros": h_c == h_g,
            "pr_auc_cpu": c5["pr_auc"], "pr_auc_gpu": g5["pr_auc"],
            "dif_pr_auc": g5["pr_auc"] - c5["pr_auc"],
            "roc_auc_cpu": c5["roc_auc"], "roc_auc_gpu": g5["roc_auc"],
            "dif_roc_auc": g5["roc_auc"] - c5["roc_auc"],
            "f2_cpu": c5["f2"], "f2_gpu": g5["f2"], "dif_f2": g5["f2"] - c5["f2"],
            "f2_optimo_cpu": co["f2"], "f2_optimo_gpu": go["f2"],
            "precision_cpu": c5["precision"], "precision_gpu": g5["precision"],
            "recall_cpu": c5["recall"], "recall_gpu": g5["recall"],
        })
    tabla = pd.DataFrame(filas)
    tabla["B1_ok"] = tabla["dif_pr_auc"].abs() <= UMBRALES["B1_dif_pr_auc"]
    tabla["pasa"] = tabla["B1_ok"]
    return tabla


def prueba_identica(variante_cpu: str) -> pd.DataFrame:
    from src.calculo_gpu.modelos_gpu import crear_modelo as crear_modelo_gpu
    from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones
    from src.referencia_cpu.modelos_cpu import crear_modelo as crear_modelo_cpu

    X, y = cargar_matrices()
    e4 = cargar_particiones()["ej4_busqueda"]
    X_ent, y_ent, X_pru, y_pru = X[e4.entrenamiento], y[e4.entrenamiento], X[e4.prueba], y[e4.prueba]
    cpu = crear_modelo_cpu("xgboost", variante_cpu)
    gpu = crear_modelo_gpu("xgboost", "gpu", calentar=False)
    filas = []
    for params in combinaciones(REJILLA_XGBOOST):
        print(f"  ajuste idéntico {params} ...", flush=True)
        p_c = np.asarray(cpu.probabilidad(cpu.ajustar(X_ent, y_ent, params), X_pru), dtype=np.float64)
        p_g = np.asarray(gpu.probabilidad(gpu.ajustar(X_ent, y_ent, params), X_pru), dtype=np.float64)
        dif = np.abs(p_g - p_c)
        pr_c, pr_g = average_precision_score(y_pru, p_c), average_precision_score(y_pru, p_g)
        filas.append({
            **params, "n_entrenamiento": len(y_ent), "n_prueba": len(y_pru),
            "max_dif_probabilidad": float(dif.max()), "media_dif_probabilidad": float(dif.mean()),
            "correlacion": float(np.corrcoef(p_c, p_g)[0, 1]),
            "filas_clase_distinta_0.5": int(np.sum((p_c >= 0.5) != (p_g >= 0.5))),
            "pr_auc_cpu": pr_c, "pr_auc_gpu": pr_g, "dif_pr_auc": pr_g - pr_c,
            "roc_auc_cpu": roc_auc_score(y_pru, p_c), "roc_auc_gpu": roc_auc_score(y_pru, p_g),
        })
    tabla = pd.DataFrame(filas)
    tabla["dif_roc_auc"] = tabla["roc_auc_gpu"] - tabla["roc_auc_cpu"]
    return tabla


# --------------------------------------------------------------------------
# Prueba extendida: GPU vs CPU, ruido natural y repetibilidad
# --------------------------------------------------------------------------
FRACCION_RUIDO = 1e-3              # submuestra que se quita del entrenamiento (0.1 %)
SEMILLAS_RUIDO = (101, 102, 103)   # fijas: la línea base es reproducible
FACTOR_COMPARABLE = 2.0            # GPU "comparable al ruido" si <= 2 x ruido máximo
NOTA_EXTENDIDA = (
    "Regla fijada el 2026-11-17 antes de correr la prueba: desviación GPU comparable al ruido "
    f"si |dif_pr_auc GPU-CPU| <= {FACTOR_COMPARABLE:g} x ruido_max_pr_auc (CPU sin 0.1 % de filas, "
    f"semillas {SEMILLAS_RUIDO}). Solo es una lectura; B1 no cambia."
)


def _tareas(exp) -> list:
    """Pares (entrenamiento, evaluación) de un experimento: 1 si holdout, 5 si cv."""
    if exp.tipo == "holdout":
        return [(exp.entrenamiento, exp.prueba)]
    return list(exp.pliegues)


def _predecir(modelo, X, y, ent, ev, params) -> np.ndarray:
    ajustado = modelo.ajustar(X[ent], y[ent], params)
    return np.asarray(modelo.probabilidad(ajustado, X[ev]), dtype=np.float64)


def _quitar_submuestra(ent: np.ndarray, fraccion: float, semilla: int) -> np.ndarray:
    """Entrenamiento sin una submuestra mínima de filas (aleatoria, semilla fija)."""
    rng = np.random.default_rng(semilla)
    mantener = np.ones(len(ent), dtype=bool)
    mantener[rng.choice(len(ent), size=max(1, round(fraccion * len(ent))), replace=False)] = False
    return ent[mantener]


def _pr_roc(y_real, p) -> tuple[float, float]:
    return float(average_precision_score(y_real, p)), float(roc_auc_score(y_real, p))


def _clases_distintas(p_a, p_b) -> int:
    return int(np.sum((p_a >= 0.5) != (p_b >= 0.5)))


def prueba_extendida(variante_cpu: str, experimentos: list[str], tabla_b1: pd.DataFrame) -> pd.DataFrame:
    from src.calculo_gpu.modelos_gpu import crear_modelo as crear_modelo_gpu
    from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones
    from src.referencia_cpu.modelos_cpu import crear_modelo as crear_modelo_cpu

    X, y = cargar_matrices()
    particiones = cargar_particiones()
    cpu = crear_modelo_cpu("xgboost", variante_cpu)
    gpu = crear_modelo_gpu("xgboost", "gpu", calentar=False)
    filas = []
    for nombre in experimentos:
        exp = particiones[nombre]
        params = json.loads(tabla_b1.loc[tabla_b1["experimento"] == nombre, "hiper_gpu"].iloc[0])
        tareas = _tareas(exp)
        print(f"  {nombre}: {params}, {len(tareas)} ajuste(s) por versión", flush=True)
        ys, p_cpu, p_a, p_b = [], [], [], []
        p_ruido = {s: [] for s in SEMILLAS_RUIDO}
        for ent, ev in tareas:
            ys.append(y[ev])
            p_cpu.append(_predecir(cpu, X, y, ent, ev, params))
            p_a.append(_predecir(gpu, X, y, ent, ev, params))
            p_b.append(_predecir(gpu, X, y, ent, ev, params))
            for s in SEMILLAS_RUIDO:
                p_ruido[s].append(_predecir(cpu, X, y, _quitar_submuestra(ent, FRACCION_RUIDO, s),
                                            ev, params))
        y_ev, c, a, b = np.concatenate(ys), np.concatenate(p_cpu), np.concatenate(p_a), np.concatenate(p_b)
        pr_c, roc_c = _pr_roc(y_ev, c)
        pr_a, roc_a = _pr_roc(y_ev, a)
        pr_b, _ = _pr_roc(y_ev, b)
        ruido = []
        for s in SEMILLAS_RUIDO:
            r = np.concatenate(p_ruido[s])
            pr_r, roc_r = _pr_roc(y_ev, r)
            ruido.append((pr_r - pr_c, roc_r - roc_c, _clases_distintas(r, c)))
        ruido_pr = max(abs(t[0]) for t in ruido)
        dif_pr = pr_a - pr_c
        filas.append({
            "experimento": nombre, "hiperparametros": json.dumps(params), "n_ajustes": len(tareas),
            "n_evaluacion": len(y_ev), "positivos_evaluacion": int(np.sum(y_ev == 1)),
            "pr_auc_cpu": pr_c, "pr_auc_gpu": pr_a, "dif_pr_auc_gpu_cpu": dif_pr,
            "dif_roc_auc_gpu_cpu": roc_a - roc_c, "filas_clase_distinta_gpu_cpu": _clases_distintas(a, c),
            "ruido_dif_pr_auc_por_semilla": json.dumps([t[0] for t in ruido]),
            "ruido_max_pr_auc": ruido_pr,
            "ruido_max_roc_auc": max(abs(t[1]) for t in ruido),
            "ruido_max_filas_clase_distinta": max(t[2] for t in ruido),
            "razon_gpu_sobre_ruido": abs(dif_pr) / ruido_pr if ruido_pr > 0 else float("inf"),
            "lectura": ("comparable al ruido" if abs(dif_pr) <= FACTOR_COMPARABLE * ruido_pr
                        else "mayor que el ruido: investigar"),
            "repet_max_dif_probabilidad": float(np.max(np.abs(a - b))),
            "repet_dif_pr_auc": pr_b - pr_a,
            "repet_filas_clase_distinta": _clases_distintas(a, b),
            "repet_identicas": bool(np.array_equal(a, b)),
        })
    return pd.DataFrame(filas)


COLUMNAS_COMPARABLES = ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc",
                        "VN", "FP", "FN", "VP"]


def comparar_corridas(otra: str) -> pd.DataFrame:
    """Métricas de la corrida actual de GPU contra las de otra carpeta (misma estructura)."""
    from pathlib import Path

    carpeta_otra = Path(otra)
    filas = []
    for exp in ORDEN_EXPERIMENTOS:
        actual = _leer("gpu", exp)
        ruta = carpeta_otra / f"parte2_xgboost_gpu_{exp}.csv"
        if not ruta.exists():
            raise SystemExit(f"Falta {ruta}")
        previa = pd.read_csv(ruta)
        if len(actual) != len(previa):
            raise SystemExit(f"{exp}: distinto número de filas ({len(actual)} vs {len(previa)})")
        dif = (actual[COLUMNAS_COMPARABLES] - previa[COLUMNAS_COMPARABLES]).abs()
        filas.append({
            "experimento": exp,
            "mismos_hiperparametros": bool((actual["hiperparametros"] == previa["hiperparametros"]).all()),
            "max_dif_pr_auc": float(dif["pr_auc"].max()), "max_dif_roc_auc": float(dif["roc_auc"].max()),
            "max_dif_f2": float(dif["f2"].max()),
            "max_dif_conteos_matriz": int(dif[["VN", "FP", "FN", "VP"]].to_numpy().max()),
            "metricas_identicas": bool((dif.to_numpy() == 0).all()),
        })
    return pd.DataFrame(filas)


_FORMATO = ["display.width", 250, "display.max_columns", None,
            "display.float_format", lambda v: f"{v:.3e}" if v != 0 and abs(v) < 1e-2 else f"{v:.6f}"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida XGBoost GPU contra XGBoost CPU (T5)")
    parser.add_argument("--cpu", default="cpu_secuencial", choices=["cpu_secuencial", "cpu_multinucleo"],
                        help="variante CPU de referencia (por defecto cpu_secuencial)")
    parser.add_argument("--extendida", action="store_true",
                        help="parte 3: GPU vs CPU con hiperparámetros idénticos, ruido y repetibilidad "
                             "en los experimentos que no pasan B1")
    parser.add_argument("--experimentos", nargs="+", default=None, choices=ORDEN_EXPERIMENTOS,
                        help="con --extendida: experimentos a probar (por defecto, los que fallan B1)")
    parser.add_argument("--comparar-con", default=None, metavar="CARPETA",
                        help="parte 4: compara los CSV de GPU actuales con los de esa carpeta")
    args = parser.parse_args()
    CARPETA.mkdir(parents=True, exist_ok=True)

    if args.comparar_con:
        tabla = comparar_corridas(args.comparar_con)
        with pd.option_context(*_FORMATO):
            print(tabla.to_string(index=False))
        ruta = CARPETA / "xgboost_gpu_repetibilidad_corridas.csv"
        tabla.to_csv(ruta, index=False)
        print(f"\nGuardado: {ruta.relative_to(RESULTADOS.parent)}")
        return

    if args.extendida:
        ruta_b1 = CARPETA / "xgboost_gpu_vs_cpu.csv"
        if not ruta_b1.exists():
            raise SystemExit("Primero corre la validación sin opciones (genera xgboost_gpu_vs_cpu.csv).")
        tabla_b1 = pd.read_csv(ruta_b1)
        pedidos = args.experimentos or [e for e in ORDEN_EXPERIMENTOS
                                        if not bool(tabla_b1.loc[tabla_b1["experimento"] == e, "B1_ok"].iloc[0])]
        print(f"3) Prueba extendida en {len(pedidos)} experimentos: {pedidos}")
        print(f"   {NOTA_EXTENDIDA}")
        tabla = prueba_extendida(args.cpu, pedidos, tabla_b1)
        with pd.option_context(*_FORMATO):
            print("\nGPU vs CPU (mismos hiperparámetros y datos) y línea base de ruido")
            print(tabla[["experimento", "dif_pr_auc_gpu_cpu", "ruido_max_pr_auc", "razon_gpu_sobre_ruido",
                         "dif_roc_auc_gpu_cpu", "ruido_max_roc_auc", "filas_clase_distinta_gpu_cpu",
                         "ruido_max_filas_clase_distinta", "lectura"]].to_string(index=False))
            print("\nRepetibilidad: dos ajustes de GPU seguidos")
            print(tabla[["experimento", "repet_max_dif_probabilidad", "repet_dif_pr_auc",
                         "repet_filas_clase_distinta", "repet_identicas"]].to_string(index=False))
        ruta = CARPETA / "xgboost_gpu_prueba_extendida.csv"
        tabla.assign(nota=NOTA_EXTENDIDA).to_csv(ruta, index=False)
        print(f"\nGuardado: {ruta.relative_to(RESULTADOS.parent)}")
        return

    print(f"1) Los 9 experimentos: XGBoost GPU contra {args.cpu} (B1: |ΔPR-AUC| <= "
          f"{UMBRALES['B1_dif_pr_auc']:g})")
    tabla = comparar_experimentos(args.cpu)
    with pd.option_context(*_FORMATO):
        print(tabla[["experimento", "hiper_cpu", "hiper_gpu", "pr_auc_cpu", "pr_auc_gpu", "dif_pr_auc",
                     "dif_roc_auc", "f2_cpu", "f2_gpu", "B1_ok"]].to_string(index=False))
    distintos = int((~tabla["mismos_hiperparametros"]).sum())
    print(f"\n   Hiperparámetros distintos entre CPU y GPU: {distintos} de {len(tabla)} experimentos "
          "(se reporta, no decide)")
    ruta = CARPETA / "xgboost_gpu_vs_cpu.csv"
    tabla.assign(umbral_B1=UMBRALES["B1_dif_pr_auc"], nota_umbrales=NOTA_UMBRALES).to_csv(ruta, index=False)
    print(f"   Guardado: {ruta.relative_to(RESULTADOS.parent)}")

    print("\n2) Prueba con hiperparámetros idénticos (4 combinaciones de la rejilla, ej4)")
    identica = prueba_identica(args.cpu)
    with pd.option_context(*_FORMATO):
        print(identica.to_string(index=False))
    ruta2 = CARPETA / "xgboost_gpu_prueba_identica.csv"
    identica.to_csv(ruta2, index=False)
    print(f"   Guardado: {ruta2.relative_to(RESULTADOS.parent)}")

    ok = bool(tabla["pasa"].all())
    print(f"\nResultado de la validación (B1 en los 9 experimentos): {'OK' if ok else 'FALLO'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
