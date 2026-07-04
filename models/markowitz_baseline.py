"""
Baseline de Markowitz (Mean-Variance Optimization) para comparación con QGNN.
Implementa la optimización clásica de la frontera eficiente.
"""
import numpy as np
from scipy.optimize import minimize
from typing import Tuple


class MarkowitzOptimizer:
    """Optimizador de carteras clásico basado en la Teoría Moderna de Carteras.

    Maximiza el Ratio de Sharpe usando la frontera eficiente.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    def optimize(self, returns_window: np.ndarray) -> np.ndarray:
        """Calcula los pesos óptimos de Markowitz para una ventana de retornos.

        Args:
            returns_window: np.ndarray de forma (T_ventana, N_activos) con retornos históricos.

        Returns:
            np.ndarray de forma (N_activos,) con los pesos óptimos (suma = 1).
        """
        n_assets = returns_window.shape[1]

        mean_returns = np.mean(returns_window, axis=0)
        cov_matrix = np.cov(returns_window.T)

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

    def predict_weights_series(self, returns_df, window: int = 60) -> list:
        """Genera una serie temporal de pesos óptimos de Markowitz,
        recalculando para cada día usando la ventana anterior.

        Args:
            returns_df: DataFrame de retornos (fechas x activos).
            window: Tamaño de la ventana histórica.

        Returns:
            Lista de arrays con pesos para cada día desde window en adelante.
        """
        weights_series = []
        returns_np = returns_df.values

        for t in range(window, len(returns_np)):
            window_data = returns_np[t - window:t]
            w = self.optimize(window_data)
            weights_series.append(w)

        return weights_series


if __name__ == "__main__":
    np.random.seed(42)
    fake_returns = np.random.randn(120, 5) * 0.02

    optimizer = MarkowitzOptimizer()
    weights = optimizer.optimize(fake_returns)
    
    print(f"Pesos óptimos Markowitz: {weights}")
    print(f"Suma de pesos: {weights.sum():.4f}")
