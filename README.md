# CC3069 — Proyecto 3: Mejorando algoritmos con CUDA

Aceleración con CUDA Python del Laboratorio 4 de CC3084 (detección de cianobacteria en los
lagos de Atitlán y Amatitlán con imágenes Sentinel-2):

- **Parte I:** índices por píxel (K1), máscara del lago y estadísticas por fecha (K2) e
  histogramas (K3) en GPU con kernels propios de Numba CUDA.
- **Parte II:** protocolo completo de validación del notebook 04 (75 ajustes de Regresión
  Logística y 93 de XGBoost) con una Regresión Logística propia en CUDA y XGBoost con
  `device="cuda"`.

Cada fase tiene tres versiones que se comparan con los mismos datos: **CPU secuencial**
(el código del laboratorio), **CPU multinúcleo** y **GPU**.

Integrantes: Angie Vela, Javier Linares, María José Yee.

## Estructura

```
lab4/                      código original del laboratorio, sin cambios (versión secuencial)
src/
├── preparacion_cpu/       FASE CPU: lectura de rásteres, dataset, particiones (nunca usa CUDA)
├── calculo_gpu/           FASE GPU: kernels y modelos (nunca lee ni escribe archivos)
├── referencia_cpu/        versiones CPU (secuencial y multinúcleo) para comparar
├── medicion/              protocolo de tiempos, monitores de CPU y GPU, ficha de la máquina
├── indices_pixel.py       fórmulas de K1 por píxel, compartidas por CPU (Numba) y GPU
├── estadisticas.py        tabla de estadísticas e histogramas, compartida por CPU y GPU
├── parametros_modelos.py  hiperparámetros compartidos de la Parte II
├── evaluacion.py          métricas del notebook 04
└── parte1_*.py, protocolo_parte2.py, validar_*.py, ...   orquestadores (leen datos, validan y miden)
notebooks/03_resultados.ipynb   tablas y gráficos del informe (solo lee los CSV de resultados/)
resultados/                tiempos, validaciones, análisis, gráficos y fichas de las máquinas
evidencias/                nvidia-smi, Administrador de Tareas, Nsight
```

La separación es estricta: `preparacion_cpu/` maneja archivos y datos y no importa CUDA;
`calculo_gpu/` recibe arreglos float32 ya preparados y no toca el disco.

## Mapa del código: versión secuencial y versiones paralelas

| Fase | CPU secuencial | CPU multinúcleo | GPU | Script que valida y mide |
| --- | --- | --- | --- | --- |
| Lectura de rásteres | `lab4/indices.leer_bandas` · `lectura.leer_cubo(modo="secuencial")` | `lectura.leer_cubo(modo="hilos")` | — (se queda en CPU) | `python -m src.preparacion_cpu.lectura` |
| K1: índices por píxel | `lab4/indices.py` | `referencia_cpu/indices_multinucleo.py` | `calculo_gpu/kernels_indices.py` | `python -m src.parte1_k1` |
| Máscara del lago + K2 | `lab4/procesar.py` | `referencia_cpu/reducciones_multinucleo.py` | `calculo_gpu/reducciones.py` | `python -m src.parte1_k2` |
| K3: histogramas | NumPy (como el notebook 02) | `referencia_cpu/reducciones_multinucleo.py` | `calculo_gpu/reducciones.py` | `python -m src.parte1_k3` |
| Dataset y particiones | `lab4/dataset_ml.py` | `preparacion_cpu/dataset.py` | — (se queda en CPU) | `python -m src.preparacion_cpu.dataset --medir` |
| Regresión Logística | scikit-learn, 1 hilo | scikit-learn, ajustes en paralelo | `calculo_gpu/regresion_logistica.py` | `python -m src.protocolo_parte2` |
| XGBoost | `tree_method="hist"`, `n_jobs=1` | `n_jobs=-1` | `calculo_gpu/xgboost_gpu.py` | `python -m src.protocolo_parte2` |

## Instalación (Windows, Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.verificar_entorno
```

`requirements.txt` fija `numpy<2.5`: `numba-cuda` 0.30.4 todavía usa `np.row_stack`, que NumPy
eliminó en la 2.5, y con NumPy 2.5 ningún kernel de Numba compila.

## Datos

`data/` no se versiona. Para descargarlos (requiere cuenta de Copernicus Data Space):

```bash
.venv\Scripts\python lab4\auth_openeo.py
.venv\Scripts\python lab4\download_sentinel.py
.venv\Scripts\python lab4\procesar.py
.venv\Scripts\python lab4\dataset_ml.py
.venv\Scripts\python -m src.preparacion_cpu.dataset
```

Si una escena falla en la descarga (tiempo de espera o "Too Many Requests"), se vuelve a correr
`download_sentinel.py`: salta las que ya existen.

### Los datos no son exactamente los del laboratorio

Copernicus ya no devuelve exactamente las mismas escenas que se descargaron para el laboratorio
(cambia sobre todo la máscara de nubes). Con la descarga actual el dataset tiene **401 793 filas y
4 866 positivos**, contra 401 656 y 4 873 del laboratorio. El proyecto evalúa la aceleración, no
reproducir las cifras del laboratorio, así que **todas las versiones (CPU y GPU) se midieron con
estos mismos datos**. Para que todos usen las mismas filas, `data/processed/matrices_ml.npz` y
`data/processed/particiones.npz` se comparten entre los integrantes en lugar de descargarlos cada
uno; sus huellas están en `resultados/validacion/particiones.csv`.

### Por qué `src.preparacion_cpu.dataset` termina con "Hay verificaciones que fallaron"

El script compara contra **cifras fijas del notebook 04 del laboratorio** (401 656 filas, 4 873
positivos, 120 497 / 1 462 en prueba, el hash del conjunto de prueba y 401 414 / 4 852 del
subconjunto espacial). Con los datos actuales esos 7 chequeos salen `[FALLO]` y el script termina
con ese mensaje. **Es lo esperado y no indica un error:**

- Los chequeos que dependen de la estructura y no de los datos salen `[ok]`: 22 escenas en la
  validación temporal, pliegues sin cruce, validación que cubre toda la base y entrenamiento y
  prueba disjuntos en los 9 experimentos.
- `dataset multinúcleo == dataset_ml.csv: True`: la versión multinúcleo da el mismo dataset.
- Las matrices y las particiones se guardan **antes** de la verificación, así que quedan listas.

Con los datos actuales, las cifras correctas son 401 793 filas, 4 866 positivos, 120 538 / 1 460
en prueba y 401 551 / 4 847 en el subconjunto espacial.

En máquinas con 16 hilos, `--medir` puede agotar la memoria de Windows ("el archivo de paginación
es demasiado pequeño") porque lanza un proceso por escena hasta el número de hilos. En ese caso:
`python -m src.preparacion_cpu.dataset --medir --procesos 6`.

## Cómo reproducir las mediciones

Todos los comandos van desde la raíz del repositorio con `.venv\Scripts\python`. Cada comparación
(secuencial, multinúcleo y GPU) se corre completa en **una misma máquina**, conectada a la
corriente y sin otros programas abiertos.

**Parte I** (medida en la máquina con RTX 3050, `resultados/maquinas/nadissa.json`):

```bash
python -m src.medicion.maquina nadissa
python -m src.referencia_cpu.parte1_secuencial   # tiempo secuencial por operación (lab4)
python -m src.preparacion_cpu.lectura            # lectura secuencial, procesos e hilos
python -m src.parte1_k1                          # K1: valida contra lab4 y mide
python -m src.parte1_k2                          # máscara del lago + K2 y Parte I completa
python -m src.parte1_k3                          # K3 y figura de histogramas
python -m src.preparacion_cpu.dataset --medir --procesos 6       # dataset secuencial vs multinúcleo
python -m src.referencia_cpu.linea_base_xgboost  # XGBoost CPU contra el HGB del laboratorio
```

La construcción del dataset y la línea base de XGBoost también se midieron en esta máquina.

**Parte II** (medida en la máquina con RTX 3060, `resultados/maquinas/yee.json`):

```bash
python -m src.protocolo_parte2 --modelo regresion_logistica --variante cpu_secuencial --maquina rtx3060
python -m src.protocolo_parte2 --modelo regresion_logistica --variante cpu_multinucleo --maquina rtx3060
python -m src.protocolo_parte2 --modelo regresion_logistica --variante gpu --maquina rtx3060
python -m src.protocolo_parte2 --modelo xgboost --variante cpu_secuencial --maquina rtx3060
python -m src.protocolo_parte2 --modelo xgboost --variante cpu_multinucleo --maquina rtx3060
python -m src.protocolo_parte2 --modelo xgboost --variante gpu --maquina rtx3060
python -m src.repetir_protocolo_parte2 --modelo xgboost --variante gpu   # 3 pasadas (desviación)
python -m src.validar_logistica_gpu               # Regresión Logística GPU contra scikit-learn
python -m src.validar_xgboost_gpu                 # XGBoost GPU contra CPU
python -m src.diagnostico_xgboost_gpu             # desglose por fase (--filas 3750000: escalabilidad)
python -m src.contar_repeticiones_parte2          # iteraciones y árboles (tabla de operaciones)
```

Tablas y gráficos del informe: `notebooks/03_resultados.ipynb` (solo lee los CSV de `resultados/`).

## Evidencias de uso del hardware

Estas corridas no modifican los CSV de tiempos:

```bash
python -m src.preparacion_cpu.lectura --evidencia 40 --modo hilos   # CPU: capturar el Administrador de Tareas
python -m src.parte1_k1 --evidencia 40                              # GPU, Parte I: registro de nvidia-smi
python -m src.evidencia_nvidia_smi --modelo xgboost                 # GPU, Parte II: registro de nvidia-smi
python -m src.perfil_nsight                                         # para Nsight Systems
```

Los registros de nvidia-smi quedan en `evidencias/nvidia-smi/` y las capturas en
`evidencias/administrador-tareas/` y `evidencias/nsight/`.
