"""Hierarchical Risk Parity (HRP) - López de Prado (2016).

Asignación de capital basada en la estructura jerárquica del grafo de
correlaciones, sin necesidad de invertir la matriz de covarianza.

Algoritmo clásico:
1) corr = cov2corr(cov); dist = sqrt((1 - corr) / 2)  (distancia de correlación)
2) link = linkage(squareform(dist, checks=False), method='single')
3) sortIx = getQuasiDiag(link)   -- orden cuasi-diagonal del dendrograma
4) w = getRecBipart(cov, sortIx) -- bisección recursiva con asignación inverse-variance
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def cov2corr(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Calcula la matriz de correlación a partir de la matriz de covarianza.

    Args:
        cov: Matriz de covarianza (N, N).
        eps: Valor mínimo para la desviación típica, evita divisiones por
            cero cuando la varianza es ~0 (defecto 1e-12).

    Returns:
        np.ndarray (N, N) con la matriz de correlación, recortada a [-1, 1].

    Notes:
        Si tras el recorte quedan NaN, se sustituyen por 0 fuera de la
        diagonal y por 1 en la diagonal (correlación de un activo consigo
        mismo).
    """
    std = np.sqrt(np.diag(cov))
    std_safe = np.where(std < eps, eps, std)
    corr = cov / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1.0, 1.0)
    if np.isnan(corr).any():
        nan_mask = np.isnan(corr)
        corr[nan_mask] = 0.0
        np.fill_diagonal(corr, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def getIVP(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance portfolio: pesos proporcionales a 1/varianza, normalizados."""
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov: np.ndarray, cItems: list) -> float:
    """Varianza de un subcluster usando pesos inverse-variance (IVP) internos."""
    cov_slice = cov[np.ix_(cItems, cItems)]
    w_ivp = getIVP(cov_slice).reshape(-1, 1)
    cluster_var = float(np.dot(w_ivp.T, np.dot(cov_slice, w_ivp))[0, 0])
    return cluster_var


def getQuasiDiag(link: np.ndarray) -> list:
    """Reordena los activos (hojas) según el dendrograma del linkage, de forma
    que activos similares queden adyacentes (orden cuasi-diagonal).

    Args:
        link: Matriz de linkage jerárquico (salida de scipy.cluster.hierarchy.linkage).

    Returns:
        Lista con el orden de los índices de activos originales.

    Notes:
        Se parte de los dos clústeres formados en la última fusión y se
        expanden iterativamente: mientras queden índices que representen
        clústeres (no hojas, es decir >= num_items), se sustituyen por sus
        dos elementos hijos hasta quedarnos solo con hojas originales.
    """
    link = link.astype(int)
    n_original_items = link[-1, 3]
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sortIx.max() >= num_items:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= num_items]
        i = df0.index
        j = df0.values - num_items
        sortIx[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = pd.concat([sortIx, df0])
        sortIx = sortIx.sort_index()
        sortIx.index = range(sortIx.shape[0])

    return sortIx.tolist()


def getRecBipart(cov: np.ndarray, sortIx: list) -> pd.Series:
    """Bisección recursiva: a cada paso se divide el cluster en dos mitades,
    se calcula la varianza inverse-variance de cada mitad y se reparte el
    peso del cluster padre de forma inversamente proporcional a esa varianza.

    Args:
        cov: Matriz de covarianza (N, N).
        sortIx: Lista con el orden cuasi-diagonal de los activos (salida de
            getQuasiDiag).

    Returns:
        pd.Series con los pesos indexados por el orden original de activos.
    """
    w = pd.Series(1.0, index=sortIx)
    cItems = [sortIx]

    while len(cItems) > 0:
        cItems = [
            c[start:stop]
            for c in cItems
            for start, stop in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]

        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]
            cItems1 = cItems[i + 1]

            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)

            alpha = 1.0 - cVar0 / (cVar0 + cVar1)

            w[cItems0] *= alpha
            w[cItems1] *= (1.0 - alpha)

    return w


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Calcula los pesos HRP (long-only, suman 1) a partir de una matriz de
    covarianza (N, N), siguiendo el algoritmo de López de Prado (2016).

    Args:
        cov: np.ndarray de forma (N, N) con la matriz de covarianza.

    Returns:
        np.ndarray de forma (N,) con pesos >= 0 que suman 1.

    Notes:
        Antes de pasar la distancia a squareform se fuerza diagonal
        exactamente cero y simetría perfecta (promediando con su
        transpuesta), ya que squareform exige una matriz de distancias
        estrictamente simétrica y con diagonal nula, condición que puede
        romperse por errores numéricos de punto flotante.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]

    if n <= 1:
        return np.array([1.0])

    corr = cov2corr(cov)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))

    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    condensed_dist = squareform(dist, checks=False)
    link = linkage(condensed_dist, method='single')

    sortIx = getQuasiDiag(link)

    w_series = getRecBipart(cov, sortIx)
    w_series = w_series.sort_index()

    weights = w_series.values.astype(float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(n) / n

    return weights


if __name__ == "__main__":
    np.random.seed(42)

    n_assets = 8
    corr = np.array([
        [1.00, 0.85, 0.80, 0.82, 0.05, 0.02, 0.04, 0.01],
        [0.85, 1.00, 0.83, 0.81, 0.03, 0.04, 0.02, 0.05],
        [0.80, 0.83, 1.00, 0.84, 0.04, 0.01, 0.03, 0.02],
        [0.82, 0.81, 0.84, 1.00, 0.02, 0.03, 0.05, 0.04],
        [0.05, 0.03, 0.04, 0.02, 1.00, 0.78, 0.75, 0.80],
        [0.02, 0.04, 0.01, 0.03, 0.78, 1.00, 0.79, 0.77],
        [0.04, 0.02, 0.03, 0.05, 0.75, 0.79, 1.00, 0.81],
        [0.01, 0.05, 0.02, 0.04, 0.80, 0.77, 0.81, 1.00],
    ])

    vols = np.array([0.10, 0.40, 0.15, 0.50, 0.12, 0.35, 0.18, 0.45])

    cov = np.outer(vols, vols) * corr

    w_hrp = hrp_weights(cov)

    print("=== Test HRP allocator ===")
    print(f"Pesos HRP: {np.round(w_hrp, 4)}")
    print(f"Forma: {w_hrp.shape} (esperado: (8,))")
    print(f"Suma de pesos: {w_hrp.sum():.6f} (esperado ~= 1)")
    print(f"Todos los pesos >= 0: {bool(np.all(w_hrp >= -1e-10))}")

    n = cov.shape[0]
    w_ew = np.ones(n) / n

    variances = np.diag(cov)
    high_var_mask = variances > np.median(variances)

    avg_w_hrp_high_var = w_hrp[high_var_mask].mean()
    avg_w_ew_high_var = w_ew[high_var_mask].mean()

    print("\n--- Coherencia inverse-variance ---")
    print(f"Varianzas por activo: {np.round(variances, 4)}")
    print(f"Peso medio HRP en activos de ALTA varianza: {avg_w_hrp_high_var:.4f}")
    print(f"Peso medio EW  en activos de ALTA varianza: {avg_w_ew_high_var:.4f}")
    print(f"HRP asigna menos peso a alta varianza que EW: "
          f"{bool(avg_w_hrp_high_var < avg_w_ew_high_var)}")

    var_hrp = float(w_hrp @ cov @ w_hrp)
    var_ew = float(w_ew @ cov @ w_ew)

    print("\n--- Varianza de cartera ---")
    print(f"Varianza cartera HRP: {var_hrp:.6f}")
    print(f"Varianza cartera EW:  {var_ew:.6f}")
    print(f"Varianza HRP < Varianza EW: {bool(var_hrp < var_ew)}")

    assert w_hrp.shape == (n_assets,), "Forma incorrecta de los pesos HRP"
    assert np.all(w_hrp >= -1e-10), "Pesos HRP negativos detectados"
    assert abs(w_hrp.sum() - 1.0) < 1e-6, "Los pesos HRP no suman 1"
    assert var_hrp < var_ew, "HRP no redujo la varianza de cartera frente a EW"

    print("\nTodas las verificaciones OK.")
