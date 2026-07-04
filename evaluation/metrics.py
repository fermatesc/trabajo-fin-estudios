"""
Métricas financieras para evaluación de carteras.
Implementa Sharpe, Sortino, Calmar, MaxDrawdown, Retorno Anualizado, etc.
"""
import sys
import os
import numpy as np
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE


def calculate_portfolio_metrics(
    daily_returns: np.ndarray,
    benchmark_returns: np.ndarray = None,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252
) -> Dict[str, float]:
    """
    Calcula un panel completo de métricas financieras.

    Args:
        daily_returns: Array 1D de retornos diarios del portafolio.
        benchmark_returns: Array 1D de retornos diarios del benchmark, misma
            longitud que daily_returns. Si se provee, se calcula el
            Information Ratio; si no, o si la longitud no coincide, queda en 0.0.
        risk_free_rate: Tasa libre de riesgo anualizada.
        trading_days: Días de trading por año (252 estándar).

    Returns:
        Diccionario con todas las métricas.

    Notes:
        Sharpe y Sortino se calculan sobre el exceso de retorno anualizado
        respecto a la tasa libre de riesgo (RISK_FREE_RATE); Sortino solo
        penaliza la volatilidad de los retornos negativos (downside risk).
        `rolling_sharpe_gt_1` es la fracción de ventanas móviles de 45 días
        (si hay al menos 45 observaciones) en las que el Sharpe rolling
        anualizado supera 1.0. Todos los ratios llevan un término 1e-8 en el
        denominador para evitar división por cero cuando la volatilidad es nula.
        `max_drawdown` es negativo (o cero); `calmar_ratio` usa su valor
        absoluto en el denominador.
    """
    daily_rf = risk_free_rate / trading_days
    excess_returns = daily_returns - daily_rf

    annualized_return = np.mean(daily_returns) * trading_days
    annualized_vol = np.std(daily_returns) * np.sqrt(trading_days)
    annualized_excess = np.mean(excess_returns) * trading_days

    sharpe = annualized_excess / (annualized_vol + 1e-8)

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(trading_days) if len(downside_returns) > 0 else 1e-8
    sortino = annualized_excess / (downside_std + 1e-8)

    cumulative = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    calmar = annualized_return / (abs(max_drawdown) + 1e-8)

    total_return = cumulative[-1] - 1.0

    if benchmark_returns is not None and len(benchmark_returns) == len(daily_returns):
        active_returns = daily_returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(trading_days)
        info_ratio = np.mean(active_returns) * trading_days / (tracking_error + 1e-8)
    else:
        info_ratio = 0.0

    rolling_sharpe_pct = 0.0
    if len(daily_returns) >= 45:
        import pandas as pd
        returns_s = pd.Series(daily_returns)
        roll_mean = returns_s.rolling(45).mean() * trading_days
        roll_std = returns_s.rolling(45).std() * np.sqrt(trading_days)
        roll_sharpe = (roll_mean - risk_free_rate) / (roll_std + 1e-8)
        rolling_sharpe_pct = (roll_sharpe > 1.0).mean()

    return {
        "retorno_anualizado": float(annualized_return),
        "volatilidad_anualizada": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
        "info_ratio": float(info_ratio),
        "rolling_sharpe_gt_1": float(rolling_sharpe_pct),
        "retorno_total": float(total_return),
        "num_dias": int(len(daily_returns))
    }

def print_metrics_table(results: Dict[str, Dict[str, float]]):
    """
    Imprime una tabla comparativa formateada de métricas por modelo.

    Args:
        results: Dict[nombre_modelo] -> Dict[metrica] -> valor
    """
    models = list(results.keys())
    metrics = ["retorno_anualizado", "volatilidad_anualizada", "sharpe_ratio",
               "sortino_ratio", "max_drawdown", "calmar_ratio", "info_ratio", "rolling_sharpe_gt_1", "retorno_total"]
    metric_names = ["Ret. Anual.", "Vol. Anual.", "Sharpe",
                    "Sortino", "Max DD", "Calmar", "Info Ratio", "Roll. Sharpe>1", "Ret. Total"]

    header = f"{'Métrica':<18}" + "".join(f"{m:>14}" for m in models)
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for metric, name in zip(metrics, metric_names):
        row = f"{name:<18}"
        for model in models:
            val = results[model].get(metric, 0)
            if "drawdown" in metric:
                row += f"{val:>13.2%} "
            elif "ratio" in metric:
                row += f"{val:>13.3f} "
            else:
                row += f"{val:>13.2%} "
        print(row)

    print("=" * len(header))
