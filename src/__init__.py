"""Paquete del Proyecto 3.

Agrega `lab4/` al `sys.path` para poder importar el código original del
Laboratorio 4 (`import config, indices, procesar, dataset_ml`) sin modificarlo.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LAB4 = RAIZ / "lab4"
RESULTADOS = RAIZ / "resultados"
EVIDENCIAS = RAIZ / "evidencias"

if str(LAB4) not in sys.path:
    sys.path.insert(0, str(LAB4))
