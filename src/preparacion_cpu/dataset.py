"""Etapa 4: dataset tabular, bloques de 1 km y particiones (CPU).

Reutiliza `dataset_ml` del Laboratorio 4. Las particiones se generan UNA vez y se
guardan como índices para que CPU y GPU entrenen exactamente con las mismas filas.
Replican, con los mismos particionadores y semillas, las del notebook 04
(`04_modelos_cianobacteria.ipynb`, rama `parte-2`, commit `6014d7a`):

    nombre                      notebook   qué es                                      ajustes
    ------------------------    --------   ----------------------------------------    -------
    ej4_busqueda                Ej. 4/5    train/test 70/30 estratificado (semilla 42)  3G + 1
                                           + StratifiedKFold(3) interno
    ej6_aleatoria               Ej. 6      StratifiedKFold(5, shuffle, 42)              5
    ej6_espacial                Ej. 6      StratifiedGroupKFold(5) por bloque_1km       5
    ej6_temporal                Ej. 6      StratifiedGroupKFold(5) por escena           5
    ej6_cronologico             Ej. 6      8 fechas antiguas / 3 recientes por lago     3G + 1
                                           + StratifiedKFold(3) interno
    ej7_atitlan_a_amatitlan     Ej. 7      entrena en un lago, evalúa en el otro        3G + 1
    ej7_amatitlan_a_atitlan     Ej. 7      (+ StratifiedKFold(3) interno)               3G + 1
    ej7_intra_atitlan           Ej. 7      70/30 estratificado dentro del lago          3G + 1
    ej7_intra_amatitlan         Ej. 7      (+ StratifiedKFold(3) interno)               3G + 1

Los experimentos ej6_* se hacen sobre el subconjunto `espacial` (bloques con al menos
50 observaciones, 401 414 filas) y usan los mejores hiperparámetros del Ej. 4, sin
búsqueda nueva. Todos los demás repiten la búsqueda de hiperparámetros.

Convención de índices (importante):

    * Todos los índices son POSICIONES GLOBALES (int32) en `X` / `y`, es decir, en
      el orden de `dataset_ml.cargar_dataset()` (0 .. 401 655).
    * Se guardan en el mismo orden en que los produce scikit-learn (por ejemplo,
      `train_test_split` devuelve el entrenamiento barajado). Ese orden se conserva
      porque el notebook entrenaba con los datos en ese orden.
    * Para usar `GridSearchCV`/`cross_val_predict` sobre `X[base]` hay que pasar los
      pliegues relativos: `pliegues_relativos(base, experimento.pliegues)`.

Contrato de las matrices (lo que consume `calculo_gpu`):

    X: np.ndarray float32 (n, 13), C-contiguo, columnas = dataset_ml.PREDICTORES
    y: np.ndarray int8 (n,), y_alta

Las 6 variables de `dataset_ml.EXCLUIDAS_POR_FUGA` nunca entran en `X`.

Uso desde la raíz del repositorio:

    .venv\\Scripts\\python -m src.preparacion_cpu.dataset
        # genera data/processed/{matrices_ml,particiones}.npz, verifica contra el
        # laboratorio y escribe resultados/validacion/particiones.csv
    .venv\\Scripts\\python -m src.preparacion_cpu.dataset --medir
        # además mide la construcción del dataset (secuencial vs multinúcleo) y la
        # preparación (CSV -> matrices, particiones)
    .venv\\Scripts\\python -m src.preparacion_cpu.dataset --evidencia
        # construye el dataset multinúcleo con registro de CPU (capturas)

Responsable: Javier Linares.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

import src  # noqa: F401  (agrega lab4/ al sys.path)
import dataset_ml as dml
from config import DATA_PROCESSED, FECHAS

SEMILLA = 42
MINIMO_POR_BLOQUE = 50
N_FECHAS_PRUEBA_CRONOLOGICO = 3

RUTA_MATRICES = DATA_PROCESSED / "matrices_ml.npz"
RUTA_PARTICIONES = DATA_PROCESSED / "particiones.npz"

# Cifras del laboratorio con las que se valida que los datos son los mismos.
FILAS_LAB = 401_656
POSITIVOS_LAB = 4_873
FILAS_ESPACIAL_LAB = 401_414
POSITIVOS_ESPACIAL_LAB = 4_852
PRUEBA_LAB = (120_497, 1_462)  # observaciones y positivos del 30 % de prueba
HASH_PRUEBA_LAB = 944216947976199151  # pd.util.hash_pandas_object(X_prueba.index).sum()

# Escenas que el Ej. 9 predice completas (mapas predictivos).
ESCENAS_MAPAS = [
    ("Amatitlan", "2026-06-19"),
    ("Amatitlan", "2026-02-02"),
    ("Atitlan", "2026-07-22"),
    ("Atitlan", "2026-04-13"),
]


# --------------------------------------------------------------------------
# Matrices
# --------------------------------------------------------------------------
def cargar_datos() -> pd.DataFrame:
    """Lee `data/processed/dataset_ml.csv` con los tipos del laboratorio."""
    return dml.cargar_dataset()


def a_matrices_float32(datos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(X float32 C-contiguo con los 13 predictores, y int8)."""
    X = np.ascontiguousarray(datos[dml.PREDICTORES].to_numpy(dtype=np.float32))
    y = np.ascontiguousarray(datos[dml.RESPUESTA].to_numpy(dtype=np.int8))
    return X, y


def guardar_matrices(X: np.ndarray, y: np.ndarray, ruta=RUTA_MATRICES) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, X=X, y=y, predictores=np.array(dml.PREDICTORES))


def cargar_matrices(ruta=RUTA_MATRICES) -> tuple[np.ndarray, np.ndarray]:
    """Lee las matrices ya generadas (mucho más rápido que releer el CSV)."""
    with np.load(ruta) as datos:
        if list(datos["predictores"]) != dml.PREDICTORES:
            raise ValueError("Las columnas guardadas no coinciden con dataset_ml.PREDICTORES")
        return np.ascontiguousarray(datos["X"]), np.ascontiguousarray(datos["y"])


def matriz_escena_float32(lago: str, fecha: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictores de TODOS los píxeles válidos de una escena (Ej. 9, mapas).

    Devuelve (X float32 C-contiguo, filas, columnas) para reinsertar la predicción
    en la malla del ráster.
    """
    X, filas, columnas = dml.matriz_escena(lago, fecha)
    return np.ascontiguousarray(X.to_numpy(dtype=np.float32)), filas, columnas


# --------------------------------------------------------------------------
# Construcción del dataset (secuencial del laboratorio vs multinúcleo)
# --------------------------------------------------------------------------
def _escenas() -> list[tuple[str, str]]:
    return [(lago, fecha) for lago, fechas in FECHAS.items() for fecha in fechas]


def _muestrear(escena: tuple[str, str]) -> pd.DataFrame:
    lago, fecha = escena
    return dml.muestrear_escena(lago, fecha)


def construir_dataset_secuencial() -> pd.DataFrame:
    """Igual que `dataset_ml.construir_dataset`, sin escribir el CSV."""
    return pd.concat([_muestrear(e) for e in _escenas()], ignore_index=True)


def construir_dataset_multinucleo(n_procesos: int | None = None) -> pd.DataFrame:
    """Misma muestra que el laboratorio, con un proceso por escena.

    `muestrear_escena` usa `random_state=42` dentro de cada escena, así que el
    resultado no depende del orden de ejecución; se concatena en el orden original.
    """
    escenas = _escenas()
    n = max(1, min(n_procesos or os.cpu_count() or 1, len(escenas)))
    with ProcessPoolExecutor(max_workers=n) as pool:
        partes = list(pool.map(_muestrear, escenas))
    return pd.concat(partes, ignore_index=True)


# --------------------------------------------------------------------------
# Particiones
# --------------------------------------------------------------------------
@dataclass
class Experimento:
    """Un experimento del notebook 04 expresado con índices globales.

    tipo "holdout": `entrenamiento` / `prueba` y `pliegues` = CV interna de 3
                    pliegues sobre `entrenamiento` (para la búsqueda de hiperparámetros).
    tipo "cv":      `base` = filas sobre las que se hace la validación cruzada y
                    `pliegues` = (entrenamiento, validación) de cada pliegue; las
                    predicciones fuera de pliegue cubren toda la `base`.
    """

    nombre: str
    tipo: str
    descripcion: str
    pliegues: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    entrenamiento: np.ndarray | None = None
    prueba: np.ndarray | None = None
    base: np.ndarray | None = None

    @property
    def ajustes_por_configuracion(self) -> int:
        return len(self.pliegues)


def _i32(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.int32))


def _cv_interna(idx_ent: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold(3, shuffle, 42) sobre `idx_ent`, como `cv_interna` del notebook."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
    y_ent = y[idx_ent]
    return [
        (_i32(idx_ent[tr]), _i32(idx_ent[va]))
        for tr, va in cv.split(np.zeros(len(idx_ent)), y_ent)
    ]


def _holdout(nombre: str, descripcion: str, idx_ent, idx_pru, y) -> Experimento:
    idx_ent, idx_pru = _i32(idx_ent), _i32(idx_pru)
    return Experimento(
        nombre=nombre, tipo="holdout", descripcion=descripcion,
        entrenamiento=idx_ent, prueba=idx_pru, pliegues=_cv_interna(idx_ent, y),
    )


def subconjunto_espacial(datos: pd.DataFrame) -> np.ndarray:
    """Posiciones de las filas en bloques de 1 km con al menos 50 observaciones."""
    conteo = datos.groupby(["lago", "bloque_1km"]).size().rename("observaciones").reset_index()
    validos = set(conteo.loc[conteo["observaciones"] >= MINIMO_POR_BLOQUE, "bloque_1km"])
    return _i32(np.flatnonzero(datos["bloque_1km"].isin(validos).to_numpy()))


def escena(datos: pd.DataFrame) -> pd.Series:
    """Identificador de escena "lago AAAA-MM-DD" (grupo de la validación temporal)."""
    return datos["lago"] + " " + datos["fecha"].dt.strftime("%Y-%m-%d")


def construir_particiones(datos: pd.DataFrame) -> dict[str, Experimento]:
    """Genera todas las particiones del §6.2 de PLAN.md (ver docstring del módulo)."""
    y = datos[dml.RESPUESTA].to_numpy(dtype=np.int8)
    n = len(datos)
    todos = np.arange(n)
    exp: dict[str, Experimento] = {}

    # Ej. 4: 70/30 estratificado. train_test_split solo depende de n, y y la semilla.
    ent, pru = train_test_split(todos, test_size=0.30, stratify=y, random_state=SEMILLA)
    exp["ej4_busqueda"] = _holdout(
        "ej4_busqueda", "Ej. 4/5: 70/30 estratificado + CV interna de 3", ent, pru, y)

    # Ej. 6: subconjunto espacial (bloques con >= 50 observaciones).
    esp = subconjunto_espacial(datos)
    y_esp = y[esp]
    sub = datos.iloc[esp]
    esquemas = {
        "ej6_aleatoria": ("Ej. 6: StratifiedKFold(5) aleatorio sobre el subconjunto espacial",
                          StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA), None),
        "ej6_espacial": ("Ej. 6: StratifiedGroupKFold(5) por bloque_1km",
                         StratifiedGroupKFold(n_splits=5), sub["bloque_1km"].to_numpy()),
        "ej6_temporal": ("Ej. 6: StratifiedGroupKFold(5) por escena (lago + fecha)",
                         StratifiedGroupKFold(n_splits=5), escena(sub).to_numpy()),
    }
    for nombre, (descripcion, cv, grupos) in esquemas.items():
        pliegues = [
            (_i32(esp[tr]), _i32(esp[va]))
            for tr, va in cv.split(np.zeros(len(esp)), y_esp, groups=grupos)
        ]
        exp[nombre] = Experimento(nombre=nombre, tipo="cv", descripcion=descripcion,
                                  base=esp, pliegues=pliegues)

    # Ej. 6: corte cronológico (3 fechas más recientes de cada lago a prueba).
    fechas_prueba = {
        f"{lago} {f}" for lago in FECHAS
        for f in sorted(FECHAS[lago])[-N_FECHAS_PRUEBA_CRONOLOGICO:]
    }
    futuro = escena(sub).isin(fechas_prueba).to_numpy()
    exp["ej6_cronologico"] = _holdout(
        "ej6_cronologico",
        "Ej. 6: 8 fechas antiguas por lago a entrenamiento, 3 recientes a prueba",
        esp[~futuro], esp[futuro], y)

    # Ej. 7: entre lagos (todo el dataset, no el subconjunto espacial).
    lagos = datos["lago"].to_numpy()
    for origen, destino in [("Atitlan", "Amatitlan"), ("Amatitlan", "Atitlan")]:
        nombre = f"ej7_{origen.lower()}_a_{destino.lower()}"
        exp[nombre] = _holdout(
            nombre, f"Ej. 7: entrena en {origen}, evalúa en {destino}",
            np.flatnonzero(lagos == origen), np.flatnonzero(lagos == destino), y)

    # Ej. 7: intra-lago, 70/30 estratificado dentro de cada lago.
    for lago in FECHAS:
        del_lago = np.flatnonzero(lagos == lago)
        ent_l, pru_l = train_test_split(
            del_lago, test_size=0.30, stratify=y[del_lago], random_state=SEMILLA)
        nombre = f"ej7_intra_{lago.lower()}"
        exp[nombre] = _holdout(nombre, f"Ej. 7: 70/30 estratificado dentro de {lago}",
                               ent_l, pru_l, y)
    return exp


def pliegues_relativos(
    base: np.ndarray, pliegues: list[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convierte pliegues globales a posiciones dentro de `X[base]`.

    Para pasar `cv=` a GridSearchCV o cross_val_predict entrenando sobre `X[base]`.
    """
    posicion = np.full(int(base.max()) + 1, -1, dtype=np.int64)
    posicion[base] = np.arange(len(base))
    relativos = []
    for ent, val in pliegues:
        r_ent, r_val = posicion[ent], posicion[val]
        if (r_ent < 0).any() or (r_val < 0).any():
            raise ValueError("Hay índices de pliegue fuera de `base`")
        relativos.append((r_ent, r_val))
    return relativos


# --------------------------------------------------------------------------
# Guardado, carga y verificación
# --------------------------------------------------------------------------
def guardar_particiones(particiones: dict[str, Experimento], ruta=RUTA_PARTICIONES) -> None:
    arreglos: dict[str, np.ndarray] = {}
    meta = []
    for e in particiones.values():
        for nombre_arreglo in ("entrenamiento", "prueba", "base"):
            valor = getattr(e, nombre_arreglo)
            if valor is not None:
                arreglos[f"{e.nombre}/{nombre_arreglo}"] = valor
        for k, (ent, val) in enumerate(e.pliegues):
            arreglos[f"{e.nombre}/pliegue{k}/entrenamiento"] = ent
            arreglos[f"{e.nombre}/pliegue{k}/validacion"] = val
        meta.append([e.nombre, e.tipo, e.descripcion, len(e.pliegues)])
    arreglos["_meta"] = np.array(meta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta, **arreglos)


def cargar_particiones(ruta=RUTA_PARTICIONES) -> dict[str, Experimento]:
    """Lee las particiones guardadas por `main()`. Úsese siempre esta función."""
    with np.load(ruta) as datos:
        particiones = {}
        for nombre, tipo, descripcion, n_pliegues in datos["_meta"]:
            def leer(clave: str):
                return datos[f"{nombre}/{clave}"] if f"{nombre}/{clave}" in datos else None

            particiones[nombre] = Experimento(
                nombre=nombre, tipo=tipo, descripcion=descripcion,
                entrenamiento=leer("entrenamiento"), prueba=leer("prueba"), base=leer("base"),
                pliegues=[
                    (datos[f"{nombre}/pliegue{k}/entrenamiento"],
                     datos[f"{nombre}/pliegue{k}/validacion"])
                    for k in range(int(n_pliegues))
                ],
            )
    return particiones


def huella(indices: np.ndarray) -> str:
    """SHA-256 corto de un arreglo de índices (para comparar entre máquinas)."""
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int32).tobytes()).hexdigest()[:16]


def hash_como_notebook(indices: np.ndarray) -> int:
    """`pd.util.hash_pandas_object(X_prueba.index).sum()` del notebook 04."""
    return int(pd.util.hash_pandas_object(pd.Index(indices.astype(np.int64))).sum())


def resumen_particiones(particiones: dict[str, Experimento], y: np.ndarray) -> pd.DataFrame:
    """Tabla con tamaño, positivos y huella de cada arreglo (se versiona)."""
    filas = []

    def fila(experimento: str, parte: str, idx: np.ndarray) -> None:
        filas.append({
            "experimento": experimento, "parte": parte, "filas": len(idx),
            "positivos": int(y[idx].sum()),
            "prevalencia_pct": round(float(y[idx].mean() * 100), 4),
            "huella_sha256": huella(idx),
        })

    for e in particiones.values():
        for parte in ("entrenamiento", "prueba", "base"):
            idx = getattr(e, parte)
            if idx is not None:
                fila(e.nombre, parte, idx)
        for k, (ent, val) in enumerate(e.pliegues):
            fila(e.nombre, f"pliegue{k}/entrenamiento", ent)
            fila(e.nombre, f"pliegue{k}/validacion", val)
    return pd.DataFrame(filas)


def verificar(datos: pd.DataFrame, particiones: dict[str, Experimento]) -> list[tuple[str, object, object, bool]]:
    """Compara contra las cifras del notebook 04. Devuelve (qué, esperado, obtenido, ok)."""
    y = datos[dml.RESPUESTA].to_numpy()
    e4, esp = particiones["ej4_busqueda"], particiones["ej6_espacial"].base
    chequeos = [
        ("filas del dataset", FILAS_LAB, len(datos)),
        ("positivos del dataset", POSITIVOS_LAB, int(y.sum())),
        ("filas de prueba (30 %)", PRUEBA_LAB[0], len(e4.prueba)),
        ("positivos de prueba", PRUEBA_LAB[1], int(y[e4.prueba].sum())),
        ("hash de X_prueba.index", HASH_PRUEBA_LAB, hash_como_notebook(e4.prueba)),
        ("filas del subconjunto espacial", FILAS_ESPACIAL_LAB, len(esp)),
        ("positivos del subconjunto espacial", POSITIVOS_ESPACIAL_LAB, int(y[esp].sum())),
        ("escenas en la validación temporal", 22, escena(datos.iloc[esp]).nunique()),
    ]
    resultado = [(q, esp_, obt, esp_ == obt) for q, esp_, obt in chequeos]

    # Cada pliegue de validación cubre la base exactamente una vez y no se cruza
    # con su entrenamiento.
    for e in particiones.values():
        for ent, val in e.pliegues:
            if np.intersect1d(ent, val).size:
                resultado.append((f"{e.nombre}: pliegue sin cruce", 0, "cruce", False))
        if e.tipo == "cv":
            union = np.sort(np.concatenate([v for _, v in e.pliegues]))
            resultado.append((f"{e.nombre}: validación cubre la base", len(e.base),
                              len(union), np.array_equal(union, np.sort(e.base))))
        else:
            cruce = np.intersect1d(e.entrenamiento, e.prueba).size
            resultado.append((f"{e.nombre}: entrenamiento y prueba disjuntos", 0, cruce, cruce == 0))
    return resultado


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------
def _preparar(datos: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Experimento]]:
    X, y = a_matrices_float32(datos)
    return X, y, construir_particiones(datos)


def _medir(n_procesos: int | None) -> None:
    from src.medicion.tiempos import guardar, medir

    n = n_procesos or os.cpu_count()
    mediciones = [
        medir(construir_dataset_secuencial, fase="Construccion del dataset (22 escenas)",
              version="CPU secuencial", repeticiones=3,
              notas="lab4 dataset_ml.muestrear_escena en bucle; sin escribir CSV; 3 corridas"),
        medir(lambda: construir_dataset_multinucleo(n),
              fase="Construccion del dataset (22 escenas)", version="CPU multinucleo",
              repeticiones=3,
              notas=f"ProcessPoolExecutor, {min(n, 22)} procesos, uno por escena; 3 corridas"),
        medir(cargar_datos, fase="Lectura de dataset_ml.csv", version="CPU secuencial"),
        medir(cargar_matrices, fase="Lectura de matrices_ml.npz", version="CPU secuencial"),
    ]
    datos = cargar_datos()
    mediciones += [
        medir(lambda: a_matrices_float32(datos), fase="DataFrame -> X/y float32",
              version="CPU secuencial"),
        medir(lambda: construir_particiones(datos), fase="Particiones (9 experimentos)",
              version="CPU secuencial"),
    ]
    for m in mediciones:
        print(f"  {m.fase:42s} {m.version:16s} {m.promedio_s:8.3f} s ± {m.desviacion_s:.3f}")
    guardar(mediciones, "parte2_preparacion.csv", sobrescribir=True)
    print("Guardado: resultados/tiempos/parte2_preparacion.csv")


def _comprobar_multinucleo(n_procesos: int | None) -> bool:
    """La construcción multinúcleo da exactamente el mismo dataset que el CSV."""
    paralelo = construir_dataset_multinucleo(n_procesos)
    referencia = cargar_datos()
    # El CSV redondea a 7 cifras significativas; se compara con esa misma precisión.
    columnas_flotantes = [c for c in referencia.columns if referencia[c].dtype == np.float32]
    iguales = (
        len(paralelo) == len(referencia)
        and (paralelo["lago"].to_numpy() == referencia["lago"].to_numpy()).all()
        and (paralelo["fila"].to_numpy() == referencia["fila"].to_numpy()).all()
        and (paralelo["columna"].to_numpy() == referencia["columna"].to_numpy()).all()
        and (paralelo[dml.RESPUESTA].to_numpy() == referencia[dml.RESPUESTA].to_numpy()).all()
        and all(
            np.allclose(paralelo[c].to_numpy(), referencia[c].to_numpy(), rtol=1e-6, atol=0)
            for c in columnas_flotantes
        )
    )
    print(f"[validación] dataset multinúcleo == dataset_ml.csv: {iguales}")
    return iguales


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrices y particiones fijas (etapa 4)")
    parser.add_argument("--medir", action="store_true", help="medir tiempos de la preparación")
    parser.add_argument("--evidencia", action="store_true",
                        help="construir el dataset multinúcleo registrando el uso de CPU")
    parser.add_argument("--procesos", type=int, default=None)
    args = parser.parse_args()

    if args.evidencia:
        from src.medicion.monitor_cpu import registrar_cpu

        print("Construyendo el dataset con todos los núcleos: capturar el Administrador de Tareas.")
        with registrar_cpu("parte2_dataset_multinucleo"):
            for _ in range(3):
                construir_dataset_multinucleo(args.procesos)
        return

    from src import RESULTADOS

    datos = cargar_datos()
    X, y, particiones = _preparar(datos)
    guardar_matrices(X, y)
    guardar_particiones(particiones)
    print(f"X {X.shape} {X.dtype}, y {y.shape} {y.dtype}  -> {RUTA_MATRICES.name}")
    print(f"{len(particiones)} experimentos                -> {RUTA_PARTICIONES.name}")

    # La carga debe devolver exactamente lo que se guardó.
    recargadas = cargar_particiones()
    assert all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        for n in particiones for a, b in zip(particiones[n].pliegues, recargadas[n].pliegues)
    )

    print("\nVerificación contra el notebook 04:")
    todo_ok = True
    for que, esperado, obtenido, ok in verificar(datos, particiones):
        todo_ok &= bool(ok)
        print(f"  [{'ok' if ok else 'FALLO'}] {que}: esperado {esperado}, obtenido {obtenido}")

    destino = RESULTADOS / "validacion"
    destino.mkdir(parents=True, exist_ok=True)
    resumen_particiones(particiones, y).to_csv(destino / "particiones.csv", index=False)
    print("\nGuardado: resultados/validacion/particiones.csv (huellas para comparar entre máquinas)")

    todo_ok &= _comprobar_multinucleo(args.procesos)
    if args.medir:
        _medir(args.procesos)
    if not todo_ok:
        raise SystemExit("Hay verificaciones que fallaron: revisar los datos antes de seguir.")


if __name__ == "__main__":
    main()
