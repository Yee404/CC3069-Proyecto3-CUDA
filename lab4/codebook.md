# Codebook

Diccionario de datos de `data/processed/dataset_ml.csv`, el conjunto tabular con el que se entrenan los modelos de clasificación de alta presencia de cianobacteria en los lagos de Atitlán y Amatitlán.

## Origen y alcance

| | |
| --- | --- |
| **Fuente** | Imágenes Sentinel-2 (productos L1C y L2A) del programa Copernicus, obtenidas vía el API openEO en la Parte 1 del laboratorio |
| **Unidad de observación** | Un píxel de 20 × 20 m dentro del espejo de agua de un lago, en una fecha de adquisición |
| **Cobertura espacial** | Lagos de Atitlán (305 114 píxeles, 122.0 km²) y Amatitlán (36 591 píxeles, 14.6 km²) |
| **Cobertura temporal** | 11 fechas por lago, entre 2025-01-18 y 2026-07-22 |
| **Proyección** | WGS 84 / UTM zona 15N (EPSG:32615), resolución 20 m |
| **Población válida** | 3,753,825 observaciones |
| **Muestra materializada** | 401,656 observaciones (10.70% de la población), muestreo aleatorio con fracción constante en las 22 escenas, `random_state = 42` |
| **Variables** | 28 |
| **Valores faltantes** | Ninguno |
| **Generado por** | `src/dataset_ml.py` (`construir_dataset`) |

## Criterios de inclusión

Una observación entra en el conjunto de datos solo si cumple las cuatro condiciones:

1. Cae dentro de `mascara_lago`, la máscara fija del espejo de agua de la Parte 1 (píxeles clasificados como agua en al menos el 60 % de las fechas).
2. Tiene `validos == True`, es decir, la banda SCL del producto L2A no la clasifica como nube, sombra de nube, cirro, píxel saturado ni sin dato.
3. Tiene valor finito en las 13 variables predictoras.
4. Tiene valor finito en `chla`, porque sin ella no hay variable respuesta definida.

Estos filtros descartan 4 930 de 3 758 755 observaciones potenciales (0.131%).

## Variable respuesta

`y_alta` vale 1 cuando la clorofila-a estimada supera **10 mg/m³**. El umbral corresponde a la frontera entre riesgo bajo y riesgo moderado para la salud humana en aguas recreativas con dominancia de cianobacteria (OMS, 2021), y es consistente con el límite del estado eutrófico de la OECD (8–25 µg/L) y con el índice de estado trófico TSI = 50 de Carlson (≈ 7.2 µg/L). Como 1 µg/L equivale a 1 mg/m³, el corte se aplica directamente sobre la columna `chla`.

Prevalencia en la población: **1.2023%** (45 133 de 3 753 825). Por lago: Amatitlán 10.9461%, Atitlán 0.0424%.

## Niveles de medición

Se usa la clasificación de Stevens: **nominal** (categorías sin orden), **ordinal** (orden sin distancia constante), **intervalo** (distancia constante, cero arbitrario) y **razón** (distancia constante y cero absoluto). Las reflectancias y las distancias son de razón; los índices normalizados son de intervalo, porque su cero no indica ausencia de la magnitud medida.

## Variables

### Identificadores y variables de agrupación

No entran en los modelos. Sirven para localizar cada observación, reconstruir los mapas predictivos y agrupar las observaciones en la validación espacial y temporal.

| Variable | Descripción | Tipo de dato | Nivel de medición | Unidades | Valores permitidos / rango | Ejemplo |
| --- | --- | --- | --- | --- | --- | --- |
| `lago` | Lago al que pertenece la observación. | `object` | Nominal | — | `{Atitlan, Amatitlan}` | `Atitlan` |
| `fecha` | Fecha de adquisición de la escena Sentinel-2 de la que proviene la observación. | `datetime64[ns]` | Intervalo | fecha (ISO 8601) | 20 valores únicos; 11 fechas por lago, 2025-01-18 a 2026-07-22 | `2025-01-18` |
| `fila` | Índice de fila del píxel dentro de la malla del ráster de su lago. | `int16` | Intervalo (índice discreto) | píxeles | Atitlán 21 – 773; Amatitlán 20 – 458 | `379` |
| `columna` | Índice de columna del píxel dentro de la malla del ráster de su lago. | `int16` | Intervalo (índice discreto) | píxeles | Atitlán 215 – 1137; Amatitlán 146 – 591 | `686` |
| `x_utm` | Coordenada este del centro del píxel, en la proyección UTM 15N. | `float64` | Intervalo | m (EPSG:32615) | Atitlán 684470 – 702910; Amatitlán 757490 – 766390 | `693890.0000` |
| `y_utm` | Coordenada norte del centro del píxel, en la proyección UTM 15N. | `float64` | Intervalo | m (EPSG:32615) | Atitlán 1616210 – 1631250; Amatitlán 1594630 – 1603390 | `1624090.0000` |
| `bloque_1km` | Bloque de la rejilla de 1 km × 1 km al que pertenece el píxel. Se usa como agrupador en la validación cruzada espacial. | `object` | Nominal | — | 205 valores (`{lago}_{x_km}_{y_km}`): 165 en Atitlán, 40 en Amatitlán | `Atitlan_693_1624` |

### Variables predictoras

Las 13 variables que reciben los modelos. Ninguna interviene, ni directa ni indirectamente, en la construcción de la variable respuesta.

| Variable | Descripción | Tipo de dato | Nivel de medición | Unidades | Valores permitidos / rango | Ejemplo |
| --- | --- | --- | --- | --- | --- | --- |
| `B02` | Reflectancia tope de atmósfera en la banda azul (490 nm) del producto Sentinel-2 L1C. | `float32` | Razón | reflectancia (0–1) | 0.0619 – 0.4777 | `0.0695` |
| `B03` | Reflectancia TOA en la banda verde (560 nm). Región donde la clorofila-a refleja más. | `float32` | Razón | reflectancia (0–1) | 0.0389 – 0.4779 | `0.0432` |
| `B07` | Reflectancia TOA en el infrarrojo cercano (783 nm). La vegetación acuática lo refleja intensamente. | `float32` | Razón | reflectancia (0–1) | 0.0102 – 0.5674 | `0.0145` |
| `B08` | Reflectancia TOA en el infrarrojo cercano (842 nm). El agua limpia lo absorbe casi por completo. | `float32` | Razón | reflectancia (0–1) | 0.0028 – 0.5626 | `0.0109` |
| `B8A` | Reflectancia TOA en el infrarrojo cercano estrecho (865 nm). | `float32` | Razón | reflectancia (0–1) | 0.0078 – 0.6146 | `0.0113` |
| `B11` | Reflectancia TOA en el infrarrojo de onda corta (1610 nm). Sensible a la humedad y al material flotante. | `float32` | Razón | reflectancia (0–1) | 0.0003 – 0.2257 | `0.0014` |
| `B12` | Reflectancia TOA en el infrarrojo de onda corta (2190 nm). Distingue agua de suelo y material seco. | `float32` | Razón | reflectancia (0–1) | 0.0000 – 0.1736 | `0.0009` |
| `ndwi` | Índice de agua de diferencia normalizada, (B03 − B08) / (B03 + B08). Heredado de la Parte 1. | `float32` | Intervalo | adimensional | -0.7147 – 0.8831 | `0.5970` |
| `mndwi` | Índice de agua modificado, (B03 − B11) / (B03 + B11). Construido para este notebook con bandas ajenas al NDCI. | `float32` | Intervalo | adimensional | -0.4027 – 0.9864 | `0.9372` |
| `ndmi` | Índice de humedad de diferencia normalizada, (B08 − B11) / (B08 + B11). Construido como sustituto limpio del NDVI. | `float32` | Intervalo | adimensional | -0.7371 – 0.9492 | `0.7724` |
| `distancia_orilla_m` | Distancia euclídea del píxel al borde del espejo de agua del lago. Proxy de profundidad y de cercanía a los aportes de la cuenca. | `float32` | Razón | m | Atitlán 20 – 3660; Amatitlán 20 – 1039 | `1241.9340` |
| `mes` | Mes de adquisición de la escena. | `int8` | Ordinal (cíclico) | mes del año | 1 – 12; presentes 1, 2, 3, 4, 5, 6, 7, 11, 12 | `1` |
| `estacion_lluviosa` | Indica si la escena se tomó en la estación lluviosa de Guatemala (mayo a octubre). | `int8` | Nominal (binaria) | — | `{0, 1}` — 0: 299,801 obs.; 1: 101,855 obs. | `0` |

### Variables excluidas por fuga de información

Están en el conjunto de datos para conservar las bandas espectrales, el NDVI, el NDWI y el índice de cianobacteria, pero **no pueden usarse como predictoras** (todas participan en la cadena que produce `y_alta`).

| Variable | Descripción | Tipo de dato | Nivel de medición | Unidades | Valores permitidos / rango | Ejemplo |
| --- | --- | --- | --- | --- | --- | --- |
| `B04` | Reflectancia TOA en la banda roja (665 nm). Es una de las dos bandas que construyen el NDCI. | `float32` | Razón | reflectancia (0–1) | 0.0158 – 0.5016 | `0.0207` |
| `B05` | Reflectancia TOA en el borde rojo (705 nm). Es el numerador del NDCI. | `float32` | Razón | reflectancia (0–1) | 0.0141 – 0.3084 | `0.0172` |
| `ndci` | Índice de clorofila de diferencia normalizada, (B05 − B04) / (B05 + B04). | `float32` | Intervalo | adimensional | -0.4052 – 0.5995 | `-0.0923` |
| `ndvi` | Índice de vegetación de diferencia normalizada, (B08 − B04) / (B08 + B04). Heredado de la Parte 1. | `float32` | Intervalo | adimensional | -0.8045 – 0.8412 | `-0.3101` |
| `fai` | Índice de algas flotantes, B07 − B04 − (B8A − B04) × 0.59. Heredado de la Parte 1. | `float32` | Intervalo | adimensional | -0.0692 – 0.2254 | `-0.0007` |
| `chla` | Clorofila-a estimada por el custom script CyanoLakes a partir del NDCI. Es la variable de la que se deriva la respuesta. | `float32` | Razón | mg/m³ | 0.0000 – 130.1181 | `0.1608` |

### Variables de contexto

Metadato de diagnóstico. No es predictor porque es constante dentro de cada escena y actuaría como identificador de fecha.

| Variable | Descripción | Tipo de dato | Nivel de medición | Unidades | Valores permitidos / rango | Ejemplo |
| --- | --- | --- | --- | --- | --- | --- |
| `nubosidad_pct` | Nubosidad de la escena completa, reportada por el catálogo de Copernicus. Constante dentro de cada escena. | `float32` | Razón | % | 0.01 – 13.00 (22 valores, uno por escena) | `0.0200` |

### Variable respuesta

El objetivo de la clasificación.

| Variable | Descripción | Tipo de dato | Nivel de medición | Unidades | Valores permitidos / rango | Ejemplo |
| --- | --- | --- | --- | --- | --- | --- |
| `y_alta` | Variable respuesta. Vale 1 si la clorofila-a del píxel supera el umbral de 10 mg/m³ de la OMS (alta presencia de cianobacteria) y 0 en caso contrario. | `int8` | Nominal (binaria) | — | `{0, 1}` — 0: 396,783 obs. (98.79%); 1: 4,873 obs. (1.21%) | `0` |

## Notas de uso

- El conjunto se lee con `dataset_ml.cargar_dataset()`, que restaura los tipos declarados en `dataset_ml.DTIPOS`. Leerlo con `pd.read_csv` sin ese mapeo devuelve todo como `float64`/`int64`.
- Las listas `dataset_ml.PREDICTORES` y `dataset_ml.EXCLUIDAS_POR_FUGA` son la definición autoritativa de qué puede entrar a un modelo. No deben solaparse.
- `fecha` tiene 20 valores únicos y no 22 porque los dos lagos comparten las fechas 2026-04-13 y 2026-04-28.
- Un mismo píxel aparece hasta 11 veces, una por fecha. Las particiones aleatorias reparten esas repeticiones entre entrenamiento y prueba, lo que infla las métricas; por eso se usa `bloque_1km` como agrupador en la validación espacial y `fecha` en la temporal.
- Los valores de `chla` provienen de un modelo calibrado con datos sintéticos para *Microcystis aeruginosa* en aguas turbias productivas (Kravitz & Matthews, 2020), con R² = 0.52 y RMSE = 84.2 mg/m³. No están validados para agua oligotrófica clara como la de Atitlán, cuya media de 1.24 mg/m³ queda por debajo del intercepto del polinomio (4.071).
