"""Repite el protocolo completo de la Parte II varias veces para medir su desviación. María José Yee.

`protocolo_parte2.py` corre el protocolo completo UNA vez por versión, así que la desviación de esa
tabla no está medida (solo las unidades representativas tienen 5 repeticiones). Este script corre
`ejecutar_protocolo` (sin modificarlo) `--pasadas` veces (3 por defecto) con el mismo modelo, las
mismas particiones y las mismas condiciones, y guarda el promedio y la desviación estándar muestral
(`statistics.stdev`, como `medicion/tiempos.py`) de cada experimento y del total.

Qué es igual que en la pasada final:
    * el mismo tiempo: el de pared de ajuste + predicción de cada experimento (`tiempo_pared_s`);
    * el mismo modelo (`_crear_modelo`, con el calentamiento de la GPU al crearlo) y los mismos
      procesos (`n_procesos` por defecto: núcleos físicos);
    * la primera pasada hace el mismo calentamiento pequeño que `protocolo_parte2.py`; las demás
      ya están calientes. Las tres cuentan en el promedio.
Qué es distinto: las métricas (precisión, ROC-AUC, F2, umbral óptimo) NO se calculan, porque no
entran en el tiempo medido y solo alargarían la corrida. Se conserva el PR-AUC de la CV interna,
que decide los hiperparámetros de los ajustes siguientes.

NO escribe en `resultados/validacion/` ni sobrescribe los CSV de tiempos finales: guarda en un CSV
nuevo por modelo y variante.

Uso (desde la raíz del repositorio):

    .venv\\Scripts\\python -m src.repetir_protocolo_parte2 --modelo xgboost --variante cpu_secuencial
    .venv\\Scripts\\python -m src.repetir_protocolo_parte2 --modelo regresion_logistica --variante gpu --espera 20

`--espera N` da N segundos antes de empezar (para iniciar una grabación del Administrador de Tareas).

Salida: resultados/tiempos/parte2_<modelo>_<variante>_<N>pasadas.csv
        (columnas de `medicion/tiempos.py`: fase, version, promedio_s, desviacion_s, corridas_s, notas)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import time

import pandas as pd

import src.protocolo_parte2 as protocolo
from src import RESULTADOS
from src.medicion.tiempos import Medicion, guardar
from src.preparacion_cpu.dataset import cargar_matrices, cargar_particiones
from src.protocolo_parte2 import ORDEN_EXPERIMENTOS, _crear_modelo, ejecutar_protocolo

# Las métricas no entran en el tiempo medido: se anulan solo en este proceso (los procesos de
# joblib de la versión multinúcleo solo ejecutan `ajustar_y_predecir`, que no las usa).
protocolo.metricas = lambda *args, **kwargs: {}
protocolo.umbral_optimo = lambda *args, **kwargs: (0.5, 0.0)


def _pitido() -> None:
    try:
        import winsound

        winsound.Beep(1000, 300)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Repite el protocolo completo para medir la desviación")
    parser.add_argument("--modelo", required=True, choices=["regresion_logistica", "xgboost"])
    parser.add_argument("--variante", required=True, choices=["cpu_secuencial", "cpu_multinucleo", "gpu"])
    parser.add_argument("--pasadas", type=int, default=3, help="pasadas completas del protocolo (3 por defecto)")
    parser.add_argument("--espera", type=int, default=0, help="segundos antes de empezar (grabación)")
    parser.add_argument("--maquina", default="rtx3060", help="etiqueta de la máquina (va en las notas)")
    args = parser.parse_args()
    if args.pasadas < 2:
        parser.error("--pasadas debe ser al menos 2 para poder calcular una desviación")

    modelo = _crear_modelo(args.modelo, args.variante)  # en GPU incluye el calentamiento
    X, y = cargar_matrices()
    particiones = cargar_particiones()
    print(f"{modelo.nombre} / {modelo.variante}: {len(X)} filas; {args.pasadas} pasadas completas del protocolo")
    print("No se guardan métricas ni se tocan los CSV finales: solo tiempos, en un CSV nuevo.")
    for s in range(args.espera, 0, -1):
        print(f"  empieza en {s} s ...", flush=True)
        time.sleep(1)
    if args.espera:
        _pitido()

    por_experimento = {e: [] for e in ORDEN_EXPERIMENTOS}
    ajustes = {}
    totales = []
    for k in range(args.pasadas):
        inicio = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):  # sin la línea por experimento
            resultado = ejecutar_protocolo(modelo, X, y, particiones, None, None, calentar_antes=(k == 0))
        pared = time.perf_counter() - inicio
        for e in ORDEN_EXPERIMENTOS:
            por_experimento[e].append(resultado["tiempos"][e]["tiempo_pared_s"])
            ajustes[e] = resultado["tiempos"][e]["n_ajustes"]
        totales.append(sum(resultado["tiempos"][e]["tiempo_pared_s"] for e in ORDEN_EXPERIMENTOS))
        print(f"  pasada {k + 1}/{args.pasadas}: protocolo {totales[-1]:.2f} s "
              f"(con la selección de hiperparámetros, {pared:.1f} s de pared)", flush=True)

    notas = f"{args.pasadas} pasadas completas; sin metricas; maquina={args.maquina}"
    mediciones = [
        Medicion(fase=f"protocolo {e}", version=modelo.version,
                 promedio_s=statistics.mean(por_experimento[e]),
                 desviacion_s=statistics.stdev(por_experimento[e]), corridas_s=por_experimento[e],
                 notas=f"{ajustes[e]} ajustes; {notas}")
        for e in ORDEN_EXPERIMENTOS
    ]
    mediciones.append(Medicion(
        fase="protocolo completo", version=modelo.version, promedio_s=statistics.mean(totales),
        desviacion_s=statistics.stdev(totales), corridas_s=totales,
        notas=f"{sum(ajustes.values())} ajustes; {notas}"))
    nombre = f"parte2_{modelo.nombre}_{modelo.variante}_{args.pasadas}pasadas.csv"
    tabla = guardar(mediciones, nombre, sobrescribir=True)

    tabla["cv_%"] = 100 * tabla["desviacion_s"] / tabla["promedio_s"]
    print("\n" + tabla[["fase", "version", "promedio_s", "desviacion_s", "cv_%"]].round(3).to_string())
    final = RESULTADOS / "tiempos" / f"parte2_{modelo.nombre}_{modelo.variante}.csv"
    if final.exists():
        previo = pd.read_csv(final)
        fila = previo[previo["fase"] == "protocolo completo"]
        if len(fila):
            print(f"\nPasada única de la corrida final (para comparar): {float(fila['promedio_s'].iloc[0]):.2f} s")
    print(f"\nGuardado: resultados/tiempos/{nombre}")
    _pitido()


if __name__ == "__main__":
    # Se importa con su nombre real para que joblib serialice las funciones por referencia.
    from src.repetir_protocolo_parte2 import main as _main

    _main()
