"""
Asignador de cartera basado en Quantum Stochastic Walk (QSW).

Implementa la caminata estocástica cuántica de Whitfield, Rodríguez-Rosario y
Aspuru-Guzik (2010, Phys. Rev. A 81, 022323) aplicada a optimización de cartera
siguiendo la formulación de Maldonado et al. (2025, npj Unconventional Computing,
arXiv:2507.03963): "Quantum Stochastic Walks for Portfolio Optimization", con el
mecanismo de personalización tipo PageRank cuántico de Sánchez-Burillo et al.
(2012, Sci. Rep. 2, 605).

Idea central
------------
Los activos son nodos de un grafo ponderado (afinidad por correlación). La QSW
generaliza la caminata aleatoria clásica y la cuántica mediante la ecuación
maestra de Lindblad:

    dρ/dt = -i (1-ω) [H, ρ]
            + ω Σ_k ( L_k ρ L_k† - ½ {L_k† L_k, ρ} )

donde ω ∈ (0,1) interpola entre dinámica coherente (cuántica) e incoherente
(caminata aleatoria clásica). Los pesos de la cartera son las POBLACIONES
(diagonal) del estado estacionario ρ_∞ que cumple 𝓛 vec(ρ)=0.

Rompiendo la trivialidad 1/N
----------------------------
La QSW canónica sobre un grafo simétrico converge al estado maximalmente mixto
(I/N → cartera equiponderada trivial), porque H conmuta con I/N. Para obtener un
"smart 1/N" no trivial seguimos el paper y la literatura de PageRank cuántico:
los operadores de salto se construyen sobre una MATRIZ DE GOOGLE columna-
estocástica con PERSONALIZACIÓN por retorno esperado,

    G = (1-g) P + g (s · 1ᵀ),     P = A diag(1/Σ_k A_kj),

donde s es el vector de personalización (≥0, suma 1) derivado del retorno/riesgo
esperado y g es la probabilidad de teletransporte (ergodicidad ⇒ estacionario
único y NO uniforme). Así el sesgo de retorno entra en el disipador y sobrevive,
y el estado estacionario refleja la centralidad del grafo ponderada por retorno.

A diferencia del VQC de la QGNN, NO hay parámetros cuánticos entrenados por
descenso de gradiente: el estado estacionario se obtiene analíticamente del núcleo
del Lindbladiano. Por construcción NO sufre barren plateaus. La calibración se
reduce a una búsqueda en rejilla de (ω, α) como en el paper.
"""
import numpy as np


def _google_matrix(A, s, teleport=0.15):
    """Construye la matriz de Google columna-estocástica con personalización.

    Args:
        A: np.ndarray (N, N) con la afinidad no negativa entre nodos.
        s: np.ndarray (N,) vector de personalización (>= 0, suma 1).
        teleport: Probabilidad de teletransporte hacia s (defecto 0.15).

    Returns:
        np.ndarray (N, N) matriz de Google columna-estocástica.

    Notes:
        Las columnas colgantes (nodo aislado, sin masa saliente) se reparten
        íntegramente según el vector de personalización, ya que de otro modo
        quedarían indefinidas tras la normalización por columna.
    """
    col_sums = A.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums <= 1e-12, 1.0, col_sums)
    P = A / col_sums
    dangling = (A.sum(axis=0) <= 1e-12)
    if dangling.any():
        P[:, dangling] = s[:, None]
    return (1.0 - teleport) * P + teleport * np.outer(s, np.ones(A.shape[0]))


def build_lindbladian(A, G, omega):
    """Construye el superoperador Lindbladiano 𝓛 (N²×N²).

    Args:
        A: np.ndarray (N, N) con la afinidad simétrica no negativa; define el
            Hamiltoniano coherente H = A.
        G: np.ndarray (N, N) matriz de transición columna-estocástica
            (Google) que define los operadores de salto.
        omega: Interpolación clásica/cuántica ω ∈ (0,1).

    Returns:
        np.ndarray complejo (N², N²) con el superoperador Lindbladiano,
        usando la convención de vectorización por columnas
        vec(X)[i + N*j] = X[i, j].

    Notes:
        El término coherente es -i(1-ω)[H,ρ] = -i(1-ω)(I⊗H - Hᵀ⊗I).
        El disipador, con operadores de salto L_ij = sqrt(G_ij)|i><j|, se
        construye en forma cerrada sin bucle explícito: el término de salto
        Σ_ij G_ij (conj(E_ij)⊗E_ij) solo puebla los índices diagonales
        (i·N+i, j·N+j) con valor G_ij, y como G es columna-estocástica
        Σ_k L_k†L_k = I, de modo que el anticonmutador se reduce a -ω·I_{N²}.
    """
    N = A.shape[0]
    H = A.astype(np.complex128)
    I = np.eye(N, dtype=np.complex128)
    coh = -1j * (1.0 - omega) * (np.kron(I, H) - np.kron(H.T, I))

    Nsq = N * N
    J = np.zeros((Nsq, Nsq), dtype=np.complex128)
    ii = np.arange(N) * (N + 1)
    J[np.ix_(ii, ii)] = G.astype(np.complex128)
    D = omega * J - omega * np.eye(Nsq, dtype=np.complex128)
    return coh + D


def stationary_populations(L):
    """Calcula las poblaciones del estado estacionario del Lindbladiano.

    Args:
        L: np.ndarray complejo (N², N²) con el superoperador Lindbladiano.

    Returns:
        np.ndarray (N,) con las poblaciones (diagonal real, no negativa y
        normalizada a suma 1) del estado estacionario ρ_∞ tal que 𝓛 vec(ρ_∞)=0.

    Notes:
        El estado estacionario se obtiene como el vector singular derecho
        asociado al valor singular más pequeño de L (núcleo del operador).
        Ese vector queda determinado salvo escala y fase complejas, que se
        fijan normalizando por la traza (traza(ρ_∞)=1); si la traza es casi
        nula (vector singular degenerado por motivos numéricos) se devuelve
        equiponderado. Tras eso se hermitiza el residuo numérico y se recorta
        la diagonal a valores no negativos antes de renormalizar.
    """
    N2 = L.shape[0]
    N = int(round(np.sqrt(N2)))
    _, _, Vh = np.linalg.svd(L)
    v = Vh[-1].conj()
    R = v.reshape(N, N, order="F")
    tr = np.trace(R)
    if abs(tr) <= 1e-12:
        return np.ones(N) / N
    rho = R / tr
    rho = 0.5 * (rho + rho.conj().T)
    pops = np.clip(np.real(np.diag(rho)), 0.0, None)
    total = pops.sum()
    return np.ones(N) / N if total <= 1e-12 else pops / total


def _personalization(N, mu, alpha):
    """Calcula el vector de personalización sesgado por retorno esperado.

    Args:
        N: Número de activos.
        mu: np.ndarray (N,) con los retornos esperados, o None para
            personalización uniforme.
        alpha: Intensidad del sesgo de retorno.

    Returns:
        np.ndarray (N,) con el vector de personalización (>= 0, suma 1).
    """
    if mu is None or alpha == 0.0:
        return np.ones(N) / N
    z = (mu - mu.mean()) / (mu.std() + 1e-8)
    s = np.exp(alpha * z)
    return s / s.sum()


def qsw_weights(corr_matrix, omega=0.5, mu=None, alpha=0.0, tau=0.5, teleport=0.15):
    """Calcula los pesos de cartera vía Quantum Stochastic Walk.

    Args:
        corr_matrix: np.ndarray (N, N) con la correlación dinámica del día.
        omega: Interpolación clásica/cuántica ω ∈ (0,1) (defecto 0.5).
        mu: np.ndarray (N,) con los retornos esperados (opcional), usado
            para la personalización.
        alpha: Intensidad del sesgo de retorno; 0 equivale a QSW puramente
            estructural (defecto 0.0).
        tau: Umbral de afinidad |corr| > tau, coherente con el grafo de la
            QGNN (defecto 0.5).
        teleport: Probabilidad de teletransporte, garantiza ergodicidad
            (defecto 0.15).

    Returns:
        np.ndarray (N,) con los pesos de cartera, long-only y que suman 1.
    """
    N = corr_matrix.shape[0]
    A = np.abs(corr_matrix).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    A[A <= tau] = 0.0
    s = _personalization(N, mu, alpha)
    G = _google_matrix(A, s, teleport=teleport)
    L = build_lindbladian(A, G, omega)
    return stationary_populations(L)
