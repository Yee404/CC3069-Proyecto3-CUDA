# Código base del Laboratorio 4 (CC3084)

Copia **sin modificaciones** de `src/` y `codebook.md` del repositorio
[Tunchxz/CC3084-Laboratorio-4](https://github.com/Tunchxz/CC3084-Laboratorio-4),
rama `parte-2`, commit `6014d7a`. Es la **solución secuencial de referencia**
del proyecto: no se edita; las versiones multinúcleo y GPU viven en `src/`.

| Archivo | Función |
| --- | --- |
| `config.py` | Bounding boxes, 22 fechas oficiales, bandas, rutas (`data/` en la raíz del repo) |
| `auth_openeo.py` | Autenticación en Copernicus Data Space (una sola vez) |
| `download_sentinel.py` | Descarga de los recortes L1C (9 bandas) y L2A (SCL) |
| `indices.py` | NDVI, NDWI, NDCI, FAI, clorofila-a y máscara de agua del script CyanoLakes (NumPy) |
| `procesar.py` | Índices por escena, máscara fija del lago y estadísticas por fecha |
| `dataset_ml.py` | Ráster → tabla, bloques de 1 km, muestreo del 10.7 % |
| `mapas.py` | Utilidades de visualización |
| `codebook.md` | Diccionario de datos del dataset de ML |

Los módulos se importan entre sí como módulos de primer nivel
(`from config import ...`); `src/__init__.py` agrega esta carpeta al `sys.path`.
