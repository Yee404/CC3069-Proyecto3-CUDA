# CC3069 — Proyecto 3: Mejorando algoritmos con CUDA

Aceleración con CUDA Python del Laboratorio 4 de CC3084 (detección de cianobacteria en los
lagos de Atitlán y Amatitlán con imágenes Sentinel-2): índices por píxel en GPU (Parte I) y
entrenamiento de Regresión Logística y XGBoost en GPU (Parte II).

Integrantes: Angie Vela, Javier Linares, María José Yee.

- Código original del laboratorio (solución secuencial de referencia): [`lab4/`](lab4/)

## Instalación (Windows, Python 3.12)

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m src.verificar_entorno
```

## Datos

`data/` no se versiona: cada integrante los descarga en su máquina (requiere cuenta de
Copernicus Data Space). Al final, `dataset_ml.csv` debe tener 401 656 filas:

```bash
.venv\Scripts\python lab4\auth_openeo.py
.venv\Scripts\python lab4\download_sentinel.py
.venv\Scripts\python lab4\procesar.py
.venv\Scripts\python lab4\dataset_ml.py
```

Después se generan las matrices y las particiones fijas que usan CPU y GPU, y se verifica
que coincidan con el laboratorio (401 656 filas, 4 873 positivos, hash del conjunto de prueba):

```bash
.venv\Scripts\python -m src.preparacion_cpu.dataset
```

Si todo sale `[ok]`, `resultados/validacion/particiones.csv` debe coincidir con la versión del
repositorio (mismas huellas en las tres máquinas).

Los comandos se ejecutan desde la raíz del repositorio.
