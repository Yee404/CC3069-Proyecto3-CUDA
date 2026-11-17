"""Regresión Logística propia en GPU (Numba CUDA + CuPy). Responsable: María José Yee.

Reproduce `StandardScaler + LogisticRegression(class_weight="balanced", C)` de scikit-learn
(solver lbfgs) con kernels propios. Este módulo NO lee ni escribe archivos.

Problema que se minimiza (el mismo que scikit-learn 1.9, `linear_model/_logistic.py`)
------------------------------------------------------------------------------------
    f(w, b) = (1 / S) * sum_i s_i * l_i(w, b)  +  ||w||^2 / (2 * C * S)

    x~_i  = (x_i - media) / escala              (StandardScaler, ddof = 0)
    z_i   = w . x~_i + b
    l_i   = log(1 + e^z_i) - y_i * z_i          (pérdida logística)
    s_i   = n / (2 * n_clase(y_i))              (class_weight="balanced")
    S     = sum_i s_i  (= n)

El intercepto `b` no se penaliza, igual que en scikit-learn. Dividir entre S no cambia el
óptimo: es la misma escala que usa scikit-learn y hace que el criterio de parada
`max|grad| <= tol` signifique lo mismo en ambos.

Decisiones (aprobadas para T4; se documentan en CONTEXTO_INFORME §11)
--------------------------------------------------------------------
* Lote completo, no mini-lotes (se aparta del PLAN §6.2). Con ~281 000 filas un ajuste
  lanza ~1 100 bloques y llena la GPU. Además es determinista y llega al mismo óptimo
  que lbfgs.
* Optimizador L-BFGS propio (memoria 10, búsqueda de línea con condiciones de Wolfe, hasta
  50 evaluaciones), con la parada de scikit-learn: max|grad| <= tol (1e-4), mejora
  relativa <= 64*eps o `max_iter` iteraciones. Arranca en w = 0, b = 0, como scikit-learn.
* Precisión: X se queda en float32 en la GPU (regla del equipo); z, la sigmoide, la
  pérdida, los acumuladores y los coeficientes van en float64, como la versión CPU.

Reparto entre GPU y CPU
-----------------------
* GPU (kernels Numba, un hilo por observación, bloques de 256):
    k_sumas_columnas     suma de cada columna y número de positivos     (media de StandardScaler)
    k_varianzas          suma de (x - media)^2 por columna              (escala, segunda pasada)
    k_perdida_gradiente  sigmoide + pérdida + gradiente ponderados       (una vez por evaluación)
    k_reducir_parciales  suma de las parciales de todos los bloques     (en orden fijo)
    k_probabilidad       P(y = 1) para la predicción
  Cada bloque reduce en memoria compartida (árbol de 256 -> 1) y escribe UNA fila de
  parciales; el segundo kernel las suma siempre en el mismo orden. No se usa
  `cuda.atomic.add`, así que dos corridas dan exactamente el mismo resultado.
* CPU (NumPy): el estado de L-BFGS son 14 números (13 coeficientes + intercepto). Por
  evaluación solo viajan 14 doubles a la GPU y 15 de vuelta; todo lo que crece con n se
  queda en la GPU.
* CuPy solo para lo auxiliar: copiar entre host y GPU, reservar memoria (su pool evita
  reservas repetidas) y aceptar arreglos que ya están en la GPU. No se usan kernels de
  CuPy (necesitan NVRTC / CUDA_PATH): Numba los ve sin copiar mediante
  `__cuda_array_interface__`.

Acceso a memoria: X es C-contiguo (n, d), así que el hilo i lee la fila i. Los hilos de un
warp leen 32 filas consecutivas (32 * 13 * 4 B = 1.6 KB contiguos) y las 13 columnas de
cada fila caen en las mismas líneas de caché.

Contrato
--------
    modelo = RegresionLogisticaGPU(C=1.0).ajustar(X, y)    # X float32 (n, d), y {0, 1}
    p = modelo.probabilidad(X_eval)                          # ndarray NumPy float64 (n,)

`X` e `y` pueden ser NumPy (se copian a la GPU dentro de `ajustar`, y esa copia entra en
el tiempo medido, como en XGBoost) o arreglos CuPy/Numba que ya están en la GPU (se usan
tal cual, sin copiarlos; deben ser C-contiguos).
"""

from __future__ import annotations

import math
from collections import deque

import cupy as cp
import numpy as np
from numba import cuda, float64

HILOS = 256                      # hilos por bloque (PLAN §6.2)
MAX_COLUMNAS = 16                # tope de predictores (el dataset usa 13)
N_COMP = MAX_COLUMNAS + 2        # gradiente (d) + intercepto + pérdida
TAM_THETA = MAX_COLUMNAS + 1     # coeficientes + intercepto
# Los tamaños de `cuda.shared.array` deben ser constantes globales: Numba no acepta una
# expresión como `MAX_COLUMNAS + 1` dentro del kernel (la ve como variable int64).
EPS = float(np.finfo(np.float64).eps)


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
@cuda.jit(device=True)
def _reducir_en_bloque(sh, n_comp, tid):
    """Reducción en árbol (256 -> 1) de las `n_comp` filas de `sh`; el resultado queda en sh[k, 0]."""
    paso = HILOS // 2
    while paso > 0:
        if tid < paso:
            for k in range(n_comp):
                sh[k, tid] += sh[k, tid + paso]
        cuda.syncthreads()
        paso //= 2


@cuda.jit
def k_sumas_columnas(X, y, parciales):
    """Por bloque: suma de cada columna de X y de y (= positivos). parciales (bloques, d + 1)."""
    sh = cuda.shared.array(shape=(N_COMP, HILOS), dtype=float64)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    n, d = X.shape
    for k in range(d + 1):
        sh[k, tid] = 0.0
    if i < n:
        for j in range(d):
            sh[j, tid] = float64(X[i, j])
        sh[d, tid] = float64(y[i])
    cuda.syncthreads()
    _reducir_en_bloque(sh, d + 1, tid)
    if tid == 0:
        for k in range(d + 1):
            parciales[cuda.blockIdx.x, k] = sh[k, 0]


@cuda.jit
def k_varianzas(X, media, parciales):
    """Por bloque: suma de (x - media)^2 por columna (segunda pasada, estable). parciales (bloques, d)."""
    sh = cuda.shared.array(shape=(N_COMP, HILOS), dtype=float64)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    n, d = X.shape
    for k in range(d):
        sh[k, tid] = 0.0
    if i < n:
        for j in range(d):
            dif = float64(X[i, j]) - media[j]
            sh[j, tid] = dif * dif
    cuda.syncthreads()
    _reducir_en_bloque(sh, d, tid)
    if tid == 0:
        for k in range(d):
            parciales[cuda.blockIdx.x, k] = sh[k, 0]


@cuda.jit
def k_perdida_gradiente(X, y, media, inv_escala, theta, peso_neg, peso_pos, parciales):
    """Sigmoide + gradiente ponderado por observación, reducido por bloque.

    theta = [w_0 .. w_{d-1}, b] (espacio estandarizado). Por bloque escribe en
    parciales[bloque, :] = [sum s*(p - y)*x~_j  (j < d),  sum s*(p - y),  sum s*l].
    Sin el 1/S ni la penalización: eso se agrega en el host, sobre 14 números.
    """
    sh = cuda.shared.array(shape=(N_COMP, HILOS), dtype=float64)
    sh_theta = cuda.shared.array(shape=TAM_THETA, dtype=float64)
    sh_media = cuda.shared.array(shape=MAX_COLUMNAS, dtype=float64)
    sh_inv = cuda.shared.array(shape=MAX_COLUMNAS, dtype=float64)
    xl = cuda.local.array(shape=MAX_COLUMNAS, dtype=float64)

    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    n, d = X.shape

    # Los parámetros son iguales para todos los hilos: se cargan una vez por bloque.
    if tid <= d:
        sh_theta[tid] = theta[tid]
    if tid < d:
        sh_media[tid] = media[tid]
        sh_inv[tid] = inv_escala[tid]
    for k in range(d + 2):
        sh[k, tid] = 0.0
    cuda.syncthreads()

    if i < n:
        z = sh_theta[d]
        for j in range(d):
            xt = (float64(X[i, j]) - sh_media[j]) * sh_inv[j]  # estandarización en registros
            xl[j] = xt
            z += sh_theta[j] * xt
        yi = float64(y[i])
        s = peso_neg + (peso_pos - peso_neg) * yi              # class_weight="balanced"
        # Sigmoide y pérdida estables: e = exp(-|z|) nunca desborda.
        e = math.exp(-abs(z))
        p = 1.0 / (1.0 + e) if z >= 0.0 else e / (1.0 + e)
        perdida = math.log1p(e) + max(z, 0.0) - yi * z         # log(1 + e^z) - y*z
        r = s * (p - yi)
        for j in range(d):
            sh[j, tid] = r * xl[j]
        sh[d, tid] = r
        sh[d + 1, tid] = s * perdida
    cuda.syncthreads()

    _reducir_en_bloque(sh, d + 2, tid)
    if tid == 0:
        for k in range(d + 2):
            parciales[cuda.blockIdx.x, k] = sh[k, 0]


@cuda.jit
def k_reducir_parciales(parciales, total):
    """Un bloque por componente k: total[k] = suma de parciales[:, k], en orden fijo."""
    sh = cuda.shared.array(shape=HILOS, dtype=float64)
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    acum = 0.0
    fila = tid
    while fila < parciales.shape[0]:
        acum += parciales[fila, k]
        fila += HILOS
    sh[tid] = acum
    cuda.syncthreads()
    paso = HILOS // 2
    while paso > 0:
        if tid < paso:
            sh[tid] += sh[tid + paso]
        cuda.syncthreads()
        paso //= 2
    if tid == 0:
        total[k] = sh[0]


@cuda.jit
def k_probabilidad(X, media, inv_escala, theta, salida):
    """salida[i] = P(y = 1 | x_i) = sigmoide(w . x~_i + b), en float64."""
    sh_theta = cuda.shared.array(shape=TAM_THETA, dtype=float64)
    sh_media = cuda.shared.array(shape=MAX_COLUMNAS, dtype=float64)
    sh_inv = cuda.shared.array(shape=MAX_COLUMNAS, dtype=float64)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    n, d = X.shape
    if tid <= d:
        sh_theta[tid] = theta[tid]
    if tid < d:
        sh_media[tid] = media[tid]
        sh_inv[tid] = inv_escala[tid]
    cuda.syncthreads()
    if i < n:
        z = sh_theta[d]
        for j in range(d):
            z += sh_theta[j] * (float64(X[i, j]) - sh_media[j]) * sh_inv[j]
        e = math.exp(-abs(z))
        salida[i] = 1.0 / (1.0 + e) if z >= 0.0 else e / (1.0 + e)


# --------------------------------------------------------------------------
# Datos en la GPU
# --------------------------------------------------------------------------
def _a_dispositivo(arreglo, dtype):
    """Arreglo de Numba en la GPU. Si ya está en la GPU no se copia; si es NumPy, se copia.

    Devuelve (arreglo_numba, dueño). El dueño (arreglo CuPy) se conserva para que la
    memoria no se libere mientras Numba la usa.
    """
    if cuda.is_cuda_array(arreglo):  # CuPy o Numba: se usa tal cual
        vista = cuda.as_cuda_array(arreglo)
        if not vista.is_c_contiguous():
            raise ValueError("El arreglo en la GPU debe ser C-contiguo")
        return vista, arreglo
    dueno = cp.asarray(np.ascontiguousarray(arreglo, dtype=dtype))  # copia CPU -> GPU
    return cuda.as_cuda_array(dueno), dueno


def _vacio(forma, dtype=np.float64):
    """Reserva en la GPU con el pool de CuPy y la expone a Numba (sin copiar)."""
    dueno = cp.empty(forma, dtype=dtype)
    return cuda.as_cuda_array(dueno), dueno


def _bloques(n: int) -> int:
    return math.ceil(n / HILOS)


def _reducir(d_parciales, d_total, n_comp: int) -> np.ndarray:
    """Suma las parciales de todos los bloques en la GPU y trae `n_comp` números al host."""
    k_reducir_parciales[n_comp, HILOS](d_parciales, d_total)
    return d_total.copy_to_host()[:n_comp]  # la copia sincroniza


class _ProblemaGPU:
    """X e y en la GPU, estadísticas de StandardScaler y evaluación de f(θ) y ∇f(θ)."""

    def __init__(self, X, y, C: float) -> None:
        self.d_X, self._dueno_X = _a_dispositivo(X, np.float32)
        self.d_y, self._dueno_y = _a_dispositivo(y, np.int8)
        n, d = self.d_X.shape
        if self.d_y.shape != (n,):
            raise ValueError(f"y debe tener forma ({n},), no {self.d_y.shape}")
        if d > MAX_COLUMNAS:
            raise ValueError(f"A lo sumo {MAX_COLUMNAS} predictores (hay {d})")
        self.n, self.d, self.C = n, d, float(C)
        self.n_bloques = _bloques(n)
        self.d_parciales, self._dueno_parciales = _vacio((self.n_bloques, d + 2))
        self.d_total, self._dueno_total = _vacio(N_COMP)
        self.d_theta, self._dueno_theta = _vacio(d + 1)

        # StandardScaler: media (1.ª pasada) y desviación con ddof = 0 (2.ª pasada).
        k_sumas_columnas[self.n_bloques, HILOS](self.d_X, self.d_y, self.d_parciales)
        sumas = _reducir(self.d_parciales, self.d_total, d + 1)
        self.media = sumas[:d] / n
        self.n_pos = int(round(sumas[d]))
        self.n_neg = n - self.n_pos
        if self.n_pos == 0 or self.n_neg == 0:
            raise ValueError("y debe tener las dos clases")
        self.d_media, self._dueno_media = _a_dispositivo(self.media, np.float64)
        k_varianzas[self.n_bloques, HILOS](self.d_X, self.d_media, self.d_parciales)
        varianza = _reducir(self.d_parciales, self.d_total, d) / n
        escala = np.sqrt(varianza)
        escala[escala == 0.0] = 1.0  # como StandardScaler con columnas constantes
        self.escala = escala
        self.d_inv, self._dueno_inv = _a_dispositivo(1.0 / escala, np.float64)

        # class_weight="balanced": n / (2 * n_clase); S = suma de pesos.
        self.peso_neg = n / (2.0 * self.n_neg)
        self.peso_pos = n / (2.0 * self.n_pos)
        self.suma_pesos = self.peso_neg * self.n_neg + self.peso_pos * self.n_pos
        self.evaluaciones = 0

    def evaluar(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """f(θ) y ∇f(θ) con θ = [w, b]. Un lanzamiento de cada kernel de pérdida y reducción."""
        d = self.d
        self.d_theta.copy_to_device(np.ascontiguousarray(theta, dtype=np.float64))
        k_perdida_gradiente[self.n_bloques, HILOS](
            self.d_X, self.d_y, self.d_media, self.d_inv, self.d_theta,
            self.peso_neg, self.peso_pos, self.d_parciales)
        total = _reducir(self.d_parciales, self.d_total, d + 2)
        self.evaluaciones += 1

        S, C = self.suma_pesos, self.C
        w = theta[:d]
        f = total[d + 1] / S + float(w @ w) / (2.0 * C * S)
        g = np.empty(d + 1)
        g[:d] = total[:d] / S + w / (C * S)
        g[d] = total[d] / S  # el intercepto no se penaliza
        return float(f), g


# --------------------------------------------------------------------------
# L-BFGS (en el host, sobre d + 1 = 14 números)
# --------------------------------------------------------------------------
def _direccion_lbfgs(g, memoria_s, memoria_y):
    """Recursión de dos ciclos: -H·g con la aproximación de L-BFGS."""
    q = g.copy()
    alfas = []
    for s, yv in reversed(list(zip(memoria_s, memoria_y))):
        rho = 1.0 / float(yv @ s)
        a = rho * float(s @ q)
        q -= a * yv
        alfas.append((rho, a, s, yv))
    if memoria_s:
        s, yv = memoria_s[-1], memoria_y[-1]
        q *= float(s @ yv) / float(yv @ yv)  # escala inicial gamma = s'y / y'y
    for rho, a, s, yv in reversed(alfas):
        b = rho * float(yv @ q)
        q += (a - b) * s
    return -q


def _busqueda_wolfe(evaluar, theta, f0, g0, direccion, paso, max_eval,
                    c1=1e-4, c2=0.9):
    """Búsqueda de línea con condiciones fuertes de Wolfe (Nocedal y Wright, alg. 3.5/3.6).

    Devuelve (paso, f, g, evaluaciones, ok). La fase de acercamiento usa bisección.
    """
    dg0 = float(g0 @ direccion)
    usadas = 0

    def probar(a):
        nonlocal usadas
        usadas += 1
        f, g = evaluar(theta + a * direccion)
        return f, g, float(g @ direccion)

    def acercar(lo, f_lo, g_lo, hi):
        mejor = (lo, f_lo, g_lo)
        while usadas < max_eval:
            a = 0.5 * (lo + hi)
            f, g, dg = probar(a)
            if f > f0 + c1 * a * dg0 or f >= f_lo:
                hi = a
            else:
                if abs(dg) <= -c2 * dg0:
                    return a, f, g, True
                if dg * (hi - lo) >= 0:
                    hi = lo
                lo, f_lo, g_lo = a, f, g
                mejor = (lo, f_lo, g_lo)
        a, f, g = mejor
        return a, f, g, a > 0  # sin Wolfe, pero con descenso de Armijo si a > 0

    a_ant, f_ant, g_ant = 0.0, f0, g0
    a = paso
    while usadas < max_eval:
        f, g, dg = probar(a)
        if f > f0 + c1 * a * dg0 or (usadas > 1 and f >= f_ant):
            a, f, g, ok = acercar(a_ant, f_ant, g_ant, a)
            return a, f, g, usadas, ok
        if abs(dg) <= -c2 * dg0:
            return a, f, g, usadas, True
        if dg >= 0:
            a, f, g, ok = acercar(a, f, g, a_ant)
            return a, f, g, usadas, ok
        a_ant, f_ant, g_ant = a, f, g
        a *= 2.0
    return a_ant, f_ant, g_ant, usadas, a_ant > 0


def lbfgs(evaluar, theta0, tol=1e-4, max_iter=1000, memoria=10, max_busqueda=50):
    """Minimiza con L-BFGS. Criterios de parada como scikit-learn (scipy L-BFGS-B):

        max|grad| <= tol              -> "gradiente"
        (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= 64 * eps   -> "mejora_relativa"
        max_iter iteraciones          -> "max_iter"
        búsqueda de línea sin avance   -> "busqueda_linea"
    """
    ftol = 64.0 * EPS
    theta = np.array(theta0, dtype=np.float64)
    f, g = evaluar(theta)
    memoria_s, memoria_y = deque(maxlen=memoria), deque(maxlen=memoria)
    motivo, iteracion = "max_iter", 0
    while iteracion < max_iter:
        if np.max(np.abs(g)) <= tol:
            motivo = "gradiente"
            break
        direccion = _direccion_lbfgs(g, memoria_s, memoria_y)
        if float(g @ direccion) >= 0.0:  # no es de descenso: se reinicia la memoria
            memoria_s.clear()
            memoria_y.clear()
            direccion = -g
        # Primer paso acotado (la dirección inicial es -g sin escalar), luego paso 1.
        paso = min(1.0, 1.0 / np.linalg.norm(g)) if not memoria_s else 1.0
        a, f_nueva, g_nueva, _, ok = _busqueda_wolfe(
            evaluar, theta, f, g, direccion, paso, max_busqueda)
        if not ok:
            motivo = "busqueda_linea"
            break
        s = a * direccion
        yv = g_nueva - g
        if float(s @ yv) > 1e-10 * float(yv @ yv):  # curvatura positiva
            memoria_s.append(s)
            memoria_y.append(yv)
        theta = theta + s
        mejora = (f - f_nueva) / max(abs(f), abs(f_nueva), 1.0)
        f, g = f_nueva, g_nueva
        iteracion += 1
        if mejora <= ftol:
            motivo = "mejora_relativa"
            break
    if motivo == "max_iter" and np.max(np.abs(g)) <= tol:
        motivo = "gradiente"
    return theta, f, g, iteracion, motivo


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
class RegresionLogisticaGPU:
    """StandardScaler + LogisticRegression(class_weight="balanced", C) con kernels propios.

    Después de `ajustar`:
        coeficientes   (d,) en el espacio estandarizado (= `coef_` de scikit-learn)
        intercepto     float (= `intercept_`)
        media, escala  (d,) (= `mean_`, `scale_` de StandardScaler)
        iteraciones, evaluaciones, perdida, max_gradiente, motivo_parada
    """

    def __init__(self, C: float = 1.0, tol: float = 1e-4, max_iter: int = 1000,
                 memoria: int = 10, max_busqueda: int = 50) -> None:
        self.C, self.tol, self.max_iter = float(C), tol, max_iter
        self.memoria, self.max_busqueda = memoria, max_busqueda

    def ajustar(self, X, y) -> "RegresionLogisticaGPU":
        problema = _ProblemaGPU(X, y, self.C)
        theta, f, g, iteraciones, motivo = lbfgs(
            problema.evaluar, np.zeros(problema.d + 1), self.tol, self.max_iter,
            self.memoria, self.max_busqueda)
        d = problema.d
        self.coeficientes, self.intercepto = theta[:d].copy(), float(theta[d])
        self.media, self.escala = problema.media, problema.escala
        self.iteraciones, self.evaluaciones = iteraciones, problema.evaluaciones
        self.perdida, self.max_gradiente = f, float(np.max(np.abs(g)))
        self.motivo_parada = motivo
        # Parámetros listos en la GPU para predecir sin volver a copiarlos.
        self._d_theta, self._dueno_theta = _a_dispositivo(theta, np.float64)
        self._d_media, self._d_inv = problema.d_media, problema.d_inv
        self._duenos = (problema._dueno_media, problema._dueno_inv)
        return self

    def probabilidad_gpu(self, X):
        """P(y = 1) como arreglo CuPy float64 que se queda en la GPU."""
        d_X, dueno_X = _a_dispositivo(X, np.float32)
        if d_X.shape[1] != len(self.coeficientes):
            raise ValueError("X no tiene el número de predictores del ajuste")
        d_salida, salida = _vacio(d_X.shape[0])
        k_probabilidad[_bloques(d_X.shape[0]), HILOS](
            d_X, self._d_media, self._d_inv, self._d_theta, d_salida)
        # `dueno_X` puede liberarse al salir: Numba y CuPy usan el mismo stream por defecto,
        # así que cualquier uso posterior de esa memoria se ejecuta después del kernel.
        return salida

    def probabilidad(self, X) -> np.ndarray:
        """P(y = 1) como ndarray NumPy float64 del host (la copia GPU -> CPU se hace aquí)."""
        return cp.asnumpy(self.probabilidad_gpu(X))


def evaluar_objetivo_gpu(X, y, C: float, theta: np.ndarray) -> tuple[float, np.ndarray]:
    """f(θ) y ∇f(θ) en un θ dado (para comprobar el kernel contra NumPy)."""
    return _ProblemaGPU(X, y, C).evaluar(np.asarray(theta, dtype=np.float64))
