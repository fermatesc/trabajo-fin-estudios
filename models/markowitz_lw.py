"""
Variante de Markowitz (Mean-Variance Optimization) con shrinkage de
Ledoit-Wolf para la estimación de la matriz de covarianza.

A diferencia de :class:`MarkowitzOptimizer` (covarianza muestral cruda
``np.cov``), esta clase estima la covarianza con
``sklearn.covariance.LedoitWolf``, que aplica un shrinkage analítico hacia
un objetivo estructurado. El resultado es una matriz mejor condicionada y
más robusta en ventanas cortas (T pequeño respecto a N), lo que suele
traducirse en pesos más estables.

La interfaz es idéntica a la de :class:`MarkowitzOptimizer`: misma firma de
``__init__``, ``optimize`` y ``predict_weights_series``, mismos bounds,
constraints y fallback a equal-weight. La ÚNICA diferencia está en cómo se
estima ``cov_matrix`` dentro de ``optimize``.

Nota: la raíz del proyecto se añade al path (independientemente del cwd)
para que la importación con prefijo de paquete (models.*) funcione tanto si
se importa como módulo del paquete como si se ejecuta el fichero directamente.
"""
import os
import sys

import numpy as np
from sklearn.covariance import LedoitWolf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.markowitz_baseline import MarkowitzOptimizer


class MarkowitzLW(MarkowitzOptimizer):
    """
    Optimizador de Markowitz que maximiza el Ratio de Sharpe usando una
    matriz de covarianza estimada con shrinkage de Ledoit-Wolf.

    Hereda de :class:`MarkowitzOptimizer` y sobrescribe únicamente
    ``optimize`` para sustituir la covarianza muestral por la estimación
    de Ledoit-Wolf. El resto del comportamiento (bounds, constraints,
    punto inicial equal-weight, fallback ante fallo de la optimización)
    permanece idéntico.
    """

    def optimize(self, returns_window: np.ndarray) -> np.ndarray:
        """Calcula los pesos óptimos de Markowitz para una ventana de retornos,
        estimando la covarianza con shrinkage de Ledoit-Wolf.

        Args:
            returns_window: np.ndarray de forma (T_ventana, N_activos) con
                retornos históricos.

        Returns:
            np.ndarray de forma (N_activos,) con los pesos óptimos (suma = 1).

        Notes:
            La única diferencia con el baseline (:class:`MarkowitzOptimizer`)
            es la estimación de la covarianza mediante shrinkage de
            Ledoit-Wolf en lugar de la covarianza muestral cruda.
        """
        from scipy.optimize import minimize

        n_assets = returns_window.shape[1]

        mean_returns = np.mean(returns_window, axis=0)
        cov_matrix = LedoitWolf().fit(returns_window).covariance_

        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if port_vol < 1e-8:
                return 1e6
            return -(port_return - self.risk_free_rate) / port_vol

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        bounds = tuple((0.0, 1.0) for _ in range(n_assets))

        w0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                neg_sharpe, w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if result.success:
                return result.x
            else:
                return w0
        except Exception:
            return w0


if __name__ == "__main__":
    np.random.seed(42)
    fake_returns = np.random.randn(120, 8) * 0.02

    base = MarkowitzOptimizer()
    lw = MarkowitzLW()

    w_base = base.optimize(fake_returns)
    w_lw = lw.optimize(fake_returns)

    for name, w in (("MarkowitzOptimizer", w_base), ("MarkowitzLW", w_lw)):
        assert np.all(w >= -1e-8), f"{name}: pesos negativos detectados -> {w}"
        assert abs(w.sum() - 1.0) < 1e-6, f"{name}: los pesos no suman 1 -> {w.sum()}"
    print("OK: ambos vectores de pesos son no-negativos y suman 1.\n")

    np.set_printoptions(precision=6, suppress=True)
    print(f"Pesos Markowitz (np.cov)      : {w_base}")
    print(f"Suma                          : {w_base.sum():.6f}\n")
    print(f"Pesos MarkowitzLW (Ledoit-Wolf): {w_lw}")
    print(f"Suma                          : {w_lw.sum():.6f}\n")

    diff = np.linalg.norm(w_base - w_lw)
    print(f"||w_base - w_lw||_2           : {diff:.6f}")
    if diff > 1e-8:
        print("El shrinkage de Ledoit-Wolf cambia la solucion respecto al baseline.")
    else:
        print("Las soluciones coinciden (diferencia despreciable).")
