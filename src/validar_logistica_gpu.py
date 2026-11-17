"""Validación de la Regresión Logística propia en GPU contra scikit-learn (T4). María José Yee.

Dos modos (desde la raíz del repositorio):

    .venv\\Scripts\\python -m src.validar_logistica_gpu --sintetico
        Prueba rápida sin archivos: kernels contra NumPy (estadísticas de StandardScaler,
        pérdida y gradiente en un θ al azar), ajustes contra scikit-learn con n que no es
        múltiplo de 256, entrada CuPy sin copia y determinismo (dos ajustes idénticos).

    .venv\\Scripts\\python -m src.validar_logistica_gpu
        Criterio adoptado (2026-11-17) en los pliegues reales de ej4_busqueda: 3 C x 3
        pliegues de la CV interna + el ajuste final con cada mejor C. CPU y GPU con
        tol=1e-4 se comparan entre sí y contra un óptimo de referencia (tol=1e-10):
            A   las referencias de scikit-learn y GPU coinciden (implementación correcta)
            B1  |PR-AUC GPU - PR-AUC CPU| <= 2e-3
            B2  pérdida GPU - pérdida del óptimo <= 2e-4
            B3  |PR-AUC GPU - PR-AUC del óptimo| <= 2e-3
            C   mejor C de la referencia, la CPU y la GPU: solo se reporta
        Los umbrales se fijaron después de ver el diagnóstico (queda anotado en el CSV).
        Salida: resultados/validacion/logistica_gpu_vs_cpu.csv

    .venv\\Scripts\\python -m src.validar_logistica_gpu --diagnostico
        Mismos ajustes de ej4 contra un óptimo de referencia con tol=1e-10 (scikit-learn
        y GPU): coincidencia de las referencias, distancia de cada versión con tol=1e-4
        al óptimo, mejor C en el óptimo verdadero y número de condición del Hessiano.
        Salida: resultados/validacion/logistica_gpu_diagnostico.csv

El modo sintético conserva su criterio propio (|ΔPR-AUC| <= 1e-3 y pérdida GPU <= CPU + 1e-6).
No mide tiempos (eso lo hace `protocolo_parte2.py` con `medicion/tiempos.py`).
Este script lee archivos, por eso vive fuera de `calculo_gpu/`.
"""

from __future__ import annotations

import argparse
import sys

import cupy as cp
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

from src import RESULTADOS
from src.calculo_gpu.modelos_gpu import crear_modelo as crear_modelo_gpu
from src.calculo_gpu.regresion_logistica import _ProblemaGPU, evaluar_objetivo_gpu
from src.parametros_modelos import REJILLA_LOGISTICA, SEMILLA
from src.referencia_cpu.modelos_cpu import crear_modelo as crear_modelo_cpu

TOL_PR_AUC = 1e-3
TOL_PERDIDA = 1e-6


# --------------------------------------------------------------------------
# Referencia en NumPy float64 (la misma para CPU y GPU)
# --------------------------------------------------------------------------
def objetivo_numpy(X, y, C, coef, intercepto, media, escala):
    """f(w, b) y ∇f de scikit-learn (lbfgs, class_weight="balanced") en float64."""
    Xs = (X.astype(np.float64) - media) / escala
    z = Xs @ coef + intercepto
    yf = y.astype(np.float64)
    n, n_pos = len(y), int(np.sum(y == 1))
    s = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * (n - n_pos)))
    S = s.sum()
    f = float(np.sum(s * (np.logaddexp(0.0, z) - yf * z)) / S + coef @ coef / (2.0 * C * S))
    r = s * (expit(z) - yf)
    g = np.concatenate([Xs.T @ r / S + coef / (C * S), [r.sum() / S]])
    return f, g


def parametros_cpu(pipe) -> dict:
    escalador, logistica = pipe[0], pipe[-1]
    return {"coef": logistica.coef_.ravel().astype(np.float64),
            "intercepto": float(logistica.intercept_[0]),
            "media": escalador.mean_, "escala": escalador.scale_,
            "iteraciones": int(logistica.n_iter_[0])}


def parametros_gpu(modelo) -> dict:
    return {"coef": modelo.coeficientes, "intercepto": modelo.intercepto,
            "media": modelo.media, "escala": modelo.escala,
            "iteraciones": modelo.iteraciones, "evaluaciones": modelo.evaluaciones,
            "motivo": modelo.motivo_parada}


def comparar(etiqueta, C, X_ent, y_ent, X_eval, y_eval, cpu, gpu) -> dict:
    """Ajusta CPU y GPU con los mismos datos y devuelve una fila de comparación."""
    m_cpu = cpu.ajustar(X_ent, y_ent, {"C": C})
    m_gpu = gpu.ajustar(X_ent, y_ent, {"C": C})
    p_cpu, p_gpu = parametros_cpu(m_cpu), parametros_gpu(m_gpu)
    f_cpu, g_cpu = objetivo_numpy(X_ent, y_ent, C, p_cpu["coef"], p_cpu["intercepto"],
                                  p_cpu["media"], p_cpu["escala"])
    f_gpu, g_gpu = objetivo_numpy(X_ent, y_ent, C, p_gpu["coef"], p_gpu["intercepto"],
                                  p_gpu["media"], p_gpu["escala"])
    pr_cpu = float(average_precision_score(y_eval, cpu.probabilidad(m_cpu, X_eval)))
    pr_gpu = float(average_precision_score(y_eval, gpu.probabilidad(m_gpu, X_eval)))
    fila = {
        "etapa": etiqueta, "C": C, "n_ent": len(y_ent), "n_eval": len(y_eval),
        "perdida_cpu": f_cpu, "perdida_gpu": f_gpu, "dif_perdida": f_gpu - f_cpu,
        "max_grad_cpu": float(np.max(np.abs(g_cpu))), "max_grad_gpu": float(np.max(np.abs(g_gpu))),
        "max_dif_coef": float(np.max(np.abs(np.append(p_gpu["coef"] - p_cpu["coef"],
                                                      p_gpu["intercepto"] - p_cpu["intercepto"])))),
        "pr_auc_cpu": pr_cpu, "pr_auc_gpu": pr_gpu, "dif_pr_auc": pr_gpu - pr_cpu,
        "iter_cpu": p_cpu["iteraciones"], "iter_gpu": p_gpu["iteraciones"],
        "eval_gpu": p_gpu["evaluaciones"], "parada_gpu": p_gpu["motivo"],
    }
    fila["pasa"] = bool(abs(fila["dif_pr_auc"]) <= TOL_PR_AUC
                        and fila["dif_perdida"] <= TOL_PERDIDA)
    return fila


def _imprimir(tabla: pd.DataFrame) -> None:
    columnas = ["etapa", "C", "dif_perdida", "max_grad_cpu", "max_grad_gpu", "max_dif_coef",
                "pr_auc_cpu", "pr_auc_gpu", "dif_pr_auc", "iter_cpu", "iter_gpu",
                "eval_gpu", "parada_gpu", "pasa"]
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", lambda v: f"{v:.3e}" if abs(v) < 1e-2 and v != 0
                           else f"{v:.6f}"):
        print(tabla[columnas].to_string(index=False))


# --------------------------------------------------------------------------
# Modo sintético (sin archivos)
# --------------------------------------------------------------------------
def datos_sinteticos(n: int, d: int = 13, semilla: int = SEMILLA):
    """Predictores correlacionados con medias y escalas distintas, ~1.2 % de positivos."""
    rng = np.random.default_rng(semilla)
    mezcla = rng.standard_normal((d, d)) * 0.4 + np.eye(d)
    Z = rng.standard_normal((n, d)) @ mezcla
    medias = rng.uniform(-0.2, 0.5, d)
    escalas = rng.uniform(0.01, 2.0, d)
    X = (Z * escalas + medias).astype(np.float32)
    beta = rng.standard_normal(d)
    puntaje = Z @ beta + rng.logistic(size=n) * 1.5  # ruido: no separable
    y = (puntaje > np.quantile(puntaje, 0.988)).astype(np.int8)
    return X, y


def correr_sintetico() -> bool:
    ok_total = True

    def revisar(nombre, error, tol):
        nonlocal ok_total
        ok = bool(error <= tol)
        ok_total &= ok
        print(f"  {'OK   ' if ok else 'FALLO'} {nombre:52s} error {error:.2e} (tol {tol:.0e})")

    X, y = datos_sinteticos(50_001)  # no es múltiplo de 256: prueba el último bloque
    print(f"Datos sintéticos: {X.shape}, {int(y.sum())} positivos")

    print("\n1) Kernels contra NumPy float64")
    problema = _ProblemaGPU(X, y, 1.0)
    X64 = X.astype(np.float64)
    media, escala = X64.mean(axis=0), X64.std(axis=0)
    # La media puede ser ~0: su error se mide relativo a la escala de cada columna.
    revisar("media (StandardScaler)", np.max(np.abs(problema.media - media) / escala), 1e-10)
    revisar("escala (StandardScaler, ddof=0)", np.max(np.abs(problema.escala - escala) / escala), 1e-10)
    theta = np.random.default_rng(1).standard_normal(X.shape[1] + 1) * 0.5
    for C in REJILLA_LOGISTICA["C"]:
        f_g, g_g = evaluar_objetivo_gpu(X, y, C, theta)
        f_n, g_n = objetivo_numpy(X, y, C, theta[:-1], theta[-1], media, escala)
        revisar(f"pérdida en θ al azar, C={C}", abs(f_g - f_n) / abs(f_n), 1e-10)
        revisar(f"gradiente en θ al azar, C={C}", np.max(np.abs(g_g - g_n)) / np.max(np.abs(g_n)), 1e-10)

    print("\n2) Ajustes contra scikit-learn (mismos datos de entrenamiento y evaluación)")
    cpu = crear_modelo_cpu("regresion_logistica", "cpu_secuencial")
    gpu = crear_modelo_gpu("regresion_logistica", "gpu", calentar=False)
    X_ev, y_ev = datos_sinteticos(20_000, semilla=SEMILLA + 1)
    filas = [comparar("sintetico", C, X, y, X_ev, y_ev, cpu, gpu) for C in REJILLA_LOGISTICA["C"]]
    tabla = pd.DataFrame(filas)
    _imprimir(tabla)
    ok_total &= bool(tabla["pasa"].all())

    print("\n3) Entrada CuPy (sin copia) y determinismo")
    m_np = gpu.ajustar(X, y, {"C": 1.0})
    m_np2 = gpu.ajustar(X, y, {"C": 1.0})
    X_cp, y_cp = cp.asarray(X), cp.asarray(y)
    m_cp = gpu.ajustar(X_cp, y_cp, {"C": 1.0})
    theta_np = np.append(m_np.coeficientes, m_np.intercepto)
    revisar("dos ajustes NumPy idénticos (bit a bit)",
            np.max(np.abs(theta_np - np.append(m_np2.coeficientes, m_np2.intercepto))), 0.0)
    revisar("ajuste con CuPy = ajuste con NumPy",
            np.max(np.abs(theta_np - np.append(m_cp.coeficientes, m_cp.intercepto))), 0.0)
    revisar("predicción con CuPy = predicción con NumPy",
            np.max(np.abs(m_np.probabilidad(X_ev) - m_np.probabilidad(cp.asarray(X_ev)))), 0.0)
    salida = m_np.probabilidad(X_ev)
    revisar("predicción: dtype float64 y forma (n,)",
            0.0 if salida.dtype == np.float64 and salida.shape == (len(X_ev),) else 1.0, 0.0)

    print(f"\nResultado de la prueba sintética: {'OK' if ok_total else 'FALLO'}")
    return ok_total


# --------------------------------------------------------------------------
# Modo ej4 (datos reales, mismos pliegues que el protocolo)
# --------------------------------------------------------------------------
# Criterio de validación adoptado para T4 (2026-11-17). A, B1, B2 y B3 deciden; el mejor C
# (C) solo se reporta. CPU y GPU se ajustan con tol=1e-4 (lo que mide el protocolo); el
# óptimo de referencia se calcula con tol=1e-10.
UMBRALES = {
    "A_dif_perdida": 1e-10,   # |pérdida ref GPU - pérdida ref CPU|
    "A_dif_coef": 1e-3,       # max |θ ref GPU - θ ref CPU|
    "A_dif_pr_auc": 1e-5,     # |PR-AUC ref GPU - PR-AUC ref CPU|
    "B1_dif_pr_auc": 2e-3,    # |PR-AUC GPU - PR-AUC CPU|            (tol=1e-4)
    "B2_exceso_perdida": 2e-4,  # pérdida GPU - pérdida del óptimo    (tol=1e-4)
    "B3_dif_pr_auc_ref": 2e-3,  # |PR-AUC GPU - PR-AUC del óptimo|    (tol=1e-4)
}
NOTA_UMBRALES = (
    "Umbrales fijados el 2026-11-17 DESPUÉS de ver el diagnóstico de ej4 "
    "(logistica_gpu_diagnostico.csv). B1-B3: scikit-learn con tol=1e-4 también los cumple "
    "(máx. observado CPU: exceso de pérdida 1.36e-4, |ΔPR-AUC vs óptimo| 1.1e-3). "
    "A: margen sobre lo observado entre las dos referencias con tol=1e-10. "
    "El mejor C se reporta pero no decide."
)


def aplicar_criterio(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega las columnas A_ok, B1_ok, B2_ok, B3_ok y pasa (A y B decisorios)."""
    u = UMBRALES
    t = tabla.copy()
    t["A_ok"] = ((t["ref_dif_perdida"].abs() <= u["A_dif_perdida"])
                 & (t["ref_max_dif_coef"] <= u["A_dif_coef"])
                 & (t["ref_dif_pr_auc"].abs() <= u["A_dif_pr_auc"]))
    t["B1_ok"] = t["dif_pr_auc_gpu_cpu"].abs() <= u["B1_dif_pr_auc"]
    t["B2_ok"] = t["gpu_exceso_perdida"] <= u["B2_exceso_perdida"]
    t["B3_ok"] = t["gpu_dif_pr_auc_ref"].abs() <= u["B3_dif_pr_auc_ref"]
    t["pasa"] = t["A_ok"] & t["B1_ok"] & t["B2_ok"] & t["B3_ok"]
    return t


def _filas_ej4():
    """Los ajustes de ej4 con referencia: 3 C x 3 pliegues + el final con cada mejor C."""
    from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones

    X, y = cargar_matrices()
    e4 = cargar_particiones()["ej4_busqueda"]
    print(f"ej4_busqueda: {len(X)} filas, {int(np.sum(y == 1))} positivos; "
          f"{len(e4.pliegues)} pliegues internos, entrenamiento {len(e4.entrenamiento)}, "
          f"prueba {len(e4.prueba)}; referencia con tol={TOL_REFERENCIA:g}")
    cpu = crear_modelo_cpu("regresion_logistica", "cpu_secuencial")
    gpu = crear_modelo_gpu("regresion_logistica", "gpu", calentar=False)

    filas = []
    for C in REJILLA_LOGISTICA["C"]:
        for k, (ent, val) in enumerate(e4.pliegues):
            print(f"  C={C:<5} pliegue {k} ...", flush=True)
            filas.append(diagnosticar(f"cv_pliegue{k}", C, X[ent], y[ent], X[val], y[val], cpu, gpu))
    cv = pd.DataFrame(filas)
    medias = cv.groupby("C")[["pr_auc_ref", "pr_auc_cpu", "pr_auc_gpu"]].mean()
    # idxmax devuelve la primera de mayor puntaje, como el protocolo.
    mejores = {c: float(medias[c].idxmax()) for c in medias.columns}
    for C in sorted(set(mejores.values())):
        print(f"  final con C={C} (entrenamiento -> prueba) ...", flush=True)
        filas.append(diagnosticar("final_prueba", C, X[e4.entrenamiento], y[e4.entrenamiento],
                                  X[e4.prueba], y[e4.prueba], cpu, gpu))
    tabla = pd.DataFrame(filas).assign(
        mejor_C_ref=mejores["pr_auc_ref"], mejor_C_cpu=mejores["pr_auc_cpu"],
        mejor_C_gpu=mejores["pr_auc_gpu"])
    return tabla, medias, mejores


_FORMATO = {"display.width": 250, "display.max_columns": None,
            "display.float_format": lambda v: f"{v:.3e}" if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4)
            else f"{v:.6f}"}


def _opciones():
    return [x for kv in _FORMATO.items() for x in kv]


def correr_ej4() -> bool:
    """Modo por defecto: criterio adoptado (A, B1, B2, B3 deciden; el mejor C se reporta)."""
    tabla, medias, mejores = _filas_ej4()
    tabla = aplicar_criterio(tabla)
    with pd.option_context(*_opciones()):
        print("\nCriterio de validación (A: referencias tol=1e-10; B: versiones tol=1e-4)")
        print(tabla[["etapa", "C", "ref_dif_perdida", "ref_max_dif_coef", "ref_dif_pr_auc",
                     "dif_pr_auc_gpu_cpu", "gpu_exceso_perdida", "gpu_dif_pr_auc_ref",
                     "A_ok", "B1_ok", "B2_ok", "B3_ok", "pasa"]].to_string(index=False))
        print("\nC (solo se reporta): PR-AUC medio de la CV interna por C")
        print(medias.to_string())
    print(f"    Mejor C -> óptimo verdadero: {mejores['pr_auc_ref']}, "
          f"CPU tol=1e-4: {mejores['pr_auc_cpu']}, GPU tol=1e-4: {mejores['pr_auc_gpu']}")
    if len(set(mejores.values())) > 1:
        print("    AVISO: el mejor C no es el mismo en las tres; no decide la validación "
              "(diferencias entre C dentro del error del optimizador con tol=1e-4)")

    carpeta = RESULTADOS / "validacion"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "logistica_gpu_vs_cpu.csv"
    salida = tabla.assign(**{f"umbral_{k}": v for k, v in UMBRALES.items()},
                          nota_umbrales=NOTA_UMBRALES)
    salida.to_csv(ruta, index=False)
    ok = bool(tabla["pasa"].all())
    print(f"\nGuardado: {ruta.relative_to(RESULTADOS.parent)}")
    print(f"Nota: {NOTA_UMBRALES}")
    print(f"Resultado de la validación en ej4: {'OK' if ok else 'FALLO'}")
    return ok


# --------------------------------------------------------------------------
# Modo diagnóstico: óptimo de referencia con tolerancia estricta
# --------------------------------------------------------------------------
TOL_REFERENCIA = 1e-10
MAX_ITER_REFERENCIA = 100_000


def _referencia_cpu(X, y, C):
    """scikit-learn con tolerancia estricta (mismo pipeline que la versión CPU)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.parametros_modelos import PARAMETROS_LOGISTICA

    params = {**PARAMETROS_LOGISTICA, "C": C, "tol": TOL_REFERENCIA,
              "max_iter": MAX_ITER_REFERENCIA}
    return make_pipeline(StandardScaler(), LogisticRegression(**params)).fit(
        X.astype(np.float64), y)


def numero_condicion(X, y, C, p) -> tuple[float, float]:
    """(cond. del Hessiano 14 x 14 de f en la solución, cond. de la correlación de X)."""
    Xs = (X.astype(np.float64) - p["media"]) / p["escala"]
    n, n_pos = len(y), int(np.sum(y == 1))
    s = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * (n - n_pos)))
    S = s.sum()
    q = expit(Xs @ p["coef"] + p["intercepto"])
    A = np.hstack([Xs, np.ones((n, 1))])
    H = (A * (s * q * (1 - q))[:, None]).T @ A / S
    H[np.arange(Xs.shape[1]), np.arange(Xs.shape[1])] += 1.0 / (C * S)  # el intercepto no
    ev_h = np.linalg.eigvalsh(H)
    ev_r = np.linalg.eigvalsh(np.corrcoef(Xs, rowvar=False))
    return float(ev_h[-1] / ev_h[0]), float(ev_r[-1] / ev_r[0])


def _theta(p) -> np.ndarray:
    return np.append(p["coef"], p["intercepto"])


def diagnosticar(etiqueta, C, X_ent, y_ent, X_eval, y_eval, cpu, gpu) -> dict:
    from src.calculo_gpu.regresion_logistica import RegresionLogisticaGPU

    versiones = {
        "cpu": (cpu, cpu.ajustar(X_ent, y_ent, {"C": C}), parametros_cpu),
        "gpu": (gpu, gpu.ajustar(X_ent, y_ent, {"C": C}), parametros_gpu),
        "ref_cpu": (cpu, _referencia_cpu(X_ent, y_ent, C), parametros_cpu),
        "ref_gpu": (gpu, RegresionLogisticaGPU(C, tol=TOL_REFERENCIA,
                                               max_iter=MAX_ITER_REFERENCIA).ajustar(X_ent, y_ent),
                    parametros_gpu),
    }
    r = {}
    for nombre, (envoltorio, modelo, extraer) in versiones.items():
        p = extraer(modelo)
        f, g = objetivo_numpy(X_ent, y_ent, C, p["coef"], p["intercepto"], p["media"], p["escala"])
        pr = float(average_precision_score(y_eval, envoltorio.probabilidad(modelo, X_eval)))
        r[nombre] = {"p": p, "f": f, "g": float(np.max(np.abs(g))), "pr": pr}

    # Óptimo de referencia = la referencia con menor pérdida.
    ref = min(("ref_cpu", "ref_gpu"), key=lambda k: r[k]["f"])
    f_ref, th_ref, pr_ref = r[ref]["f"], _theta(r[ref]["p"]), r[ref]["pr"]
    cond_h, cond_r = numero_condicion(X_ent, y_ent, C, r[ref]["p"])
    fila = {
        "etapa": etiqueta, "C": C, "referencia": ref,
        # (1) ¿coinciden las dos referencias?
        "ref_dif_perdida": r["ref_gpu"]["f"] - r["ref_cpu"]["f"],
        "ref_max_dif_coef": float(np.max(np.abs(_theta(r["ref_gpu"]["p"]) - _theta(r["ref_cpu"]["p"])))),
        "ref_dif_pr_auc": r["ref_gpu"]["pr"] - r["ref_cpu"]["pr"],
        "ref_max_grad_cpu": r["ref_cpu"]["g"], "ref_max_grad_gpu": r["ref_gpu"]["g"],
        "ref_iter_cpu": r["ref_cpu"]["p"]["iteraciones"], "ref_iter_gpu": r["ref_gpu"]["p"]["iteraciones"],
        "ref_parada_gpu": r["ref_gpu"]["p"]["motivo"],
        "pr_auc_ref": pr_ref,
        # números de condición
        "cond_hessiano": cond_h, "cond_correlacion_X": cond_r,
    }
    # (2) distancia de cada versión con tol=1e-4 al óptimo
    for v in ("cpu", "gpu"):
        fila[f"{v}_exceso_perdida"] = r[v]["f"] - f_ref
        fila[f"{v}_dist_coef"] = float(np.max(np.abs(_theta(r[v]["p"]) - th_ref)))
        fila[f"{v}_dif_pr_auc_ref"] = r[v]["pr"] - pr_ref
        fila[f"pr_auc_{v}"] = r[v]["pr"]
    fila["dif_pr_auc_gpu_cpu"] = r["gpu"]["pr"] - r["cpu"]["pr"]
    return fila


def correr_diagnostico() -> bool:
    """Modo aparte: todas las columnas del diagnóstico, sin criterio de aprobación."""
    print(f"Diagnóstico ej4: max_iter de la referencia = {MAX_ITER_REFERENCIA}")
    tabla, medias, mejores = _filas_ej4()
    with pd.option_context(*_opciones()):
        print("\n(1) ¿Coinciden scikit-learn y GPU con tol=1e-10?")
        print(tabla[["etapa", "C", "ref_dif_perdida", "ref_max_dif_coef", "ref_dif_pr_auc",
                     "ref_max_grad_cpu", "ref_max_grad_gpu", "ref_iter_cpu", "ref_iter_gpu",
                     "ref_parada_gpu"]].to_string(index=False))
        print("\n(2) Distancia de cada versión con tol=1e-4 al óptimo de referencia")
        print(tabla[["etapa", "C", "cpu_exceso_perdida", "gpu_exceso_perdida", "cpu_dist_coef",
                     "gpu_dist_coef", "cpu_dif_pr_auc_ref", "gpu_dif_pr_auc_ref",
                     "dif_pr_auc_gpu_cpu"]].to_string(index=False))
        print("\n(3) PR-AUC medio de la CV interna por C")
        print(medias.to_string())
        print(f"    Mejor C -> óptimo verdadero: {mejores['pr_auc_ref']}, "
              f"CPU tol=1e-4: {mejores['pr_auc_cpu']}, GPU tol=1e-4: {mejores['pr_auc_gpu']}")
        print("\n(4) Números de condición")
        print(tabla[["etapa", "C", "cond_hessiano", "cond_correlacion_X"]].to_string(index=False))

    carpeta = RESULTADOS / "validacion"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "logistica_gpu_diagnostico.csv"
    tabla.to_csv(ruta, index=False)
    print(f"\nGuardado: {ruta.relative_to(RESULTADOS.parent)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida la Regresión Logística GPU (T4)")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--sintetico", action="store_true",
                      help="prueba rápida con datos sintéticos (sin leer archivos)")
    modo.add_argument("--diagnostico", action="store_true",
                      help="ej4 contra un óptimo de referencia con tol=1e-10")
    args = parser.parse_args()
    if args.sintetico:
        ok = correr_sintetico()
    elif args.diagnostico:
        ok = correr_diagnostico()
    else:
        ok = correr_ej4()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
