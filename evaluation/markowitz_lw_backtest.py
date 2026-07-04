"""
Baseline adicional para la evaluacion principal: Markowitz con shrinkage de
Ledoit-Wolf (MarkowitzLW), bajo EXACTAMENTE el mismo protocolo que
``evaluation/backtesting.py`` produce para el Markowitz crudo y el Equal-Weight
(mismo universo de 20 activos, mismo tramo de test 2023, ventana movil de 60
jornadas, coste de 10 pb sobre la rotacion con contabilidad drift-consciente y
tasa libre de riesgo del 4,5%).

Motivacion (revision por pares): el Cap. 2 identifica el shrinkage (Ledoit-Wolf)
y Black-Litterman como las correcciones estandar a la inestabilidad de
media-varianza, pero el unico Markowitz de la Tabla principal es la version
ingenua de covarianza muestral. Este script anade la version estabilizada para
que el baseline clasico no sea un hombre de paja.

No reentrena ni toca los modelos neuronales: solo recalcula el bloque clasico.
El Equal-Weight se recomputa identicamente para usarlo como benchmark del
Information Ratio (igual que en backtesting.py).

Uso:
    python -m evaluation.markowitz_lw_backtest
Salida:
    markowitz_lw_metrics.json
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.dataset_loader import FinancialGraphDataset
from evaluation.metrics import calculate_portfolio_metrics
from models.markowitz_baseline import MarkowitzOptimizer
from models.markowitz_lw import MarkowitzLW
from config import TRANSACTION_COST_BPS


def _run_allocator(allocator, dataset, test_indices):
    """Backtest drift-consciente de un asignador con interfaz .optimize(window)."""
    N = dataset.num_assets
    ew = np.ones(N) / N
    w_prev = ew.copy()
    rets, turns = [], []
    for i in test_indices:
        data = dataset[i]
        idx_date = dataset.valid_dates[i]
        window_returns = dataset.returns.loc[:idx_date].iloc[-60:].values
        w = allocator.optimize(window_returns)
        y_simple = np.expm1(data.y.numpy())
        turnover = np.abs(w - w_prev).sum()
        ret = np.dot(w, y_simple) - (turnover * TRANSACTION_COST_BPS / 10000.0)
        rets.append(float(ret))
        turns.append(float(turnover))
        w_drift = w * (1.0 + y_simple)
        w_prev = w_drift / w_drift.sum()
    return np.array(rets), float(np.mean(turns))


def _run_ew(dataset, test_indices):
    """Backtest drift-consciente de la cartera Equal-Weight (mismo protocolo que _run_allocator)."""
    N = dataset.num_assets
    ew = np.ones(N) / N
    w_prev = ew.copy()
    rets, turns = [], []
    for i in test_indices:
        data = dataset[i]
        y_simple = np.expm1(data.y.numpy())
        turnover = np.abs(ew - w_prev).sum()
        ret = np.dot(ew, y_simple) - (turnover * TRANSACTION_COST_BPS / 10000.0)
        rets.append(float(ret))
        turns.append(float(turnover))
        w_drift = ew * (1.0 + y_simple)
        w_prev = w_drift / w_drift.sum()
    return np.array(rets), float(np.mean(turns))


def run():
    """Ejecuta el backtest de Equal-Weight, Markowitz y Markowitz-LW y guarda
    las métricas comparativas en markowitz_lw_metrics.json."""
    dataset = FinancialGraphDataset(tau=0.5)
    total = len(dataset)
    train_size = int(total * 0.70)
    test_indices = list(range(train_size, total))
    print(f"Universo: {dataset.num_assets} activos | test: {len(test_indices)} jornadas")

    ew_rets, ew_turn = _run_ew(dataset, test_indices)
    mk_rets, mk_turn = _run_allocator(MarkowitzOptimizer(), dataset, test_indices)
    lw_rets, lw_turn = _run_allocator(MarkowitzLW(), dataset, test_indices)

    out = {}
    for name, rets, turn in [
        ("Equal_Weight", ew_rets, ew_turn),
        ("Markowitz", mk_rets, mk_turn),
        ("Markowitz_LW", lw_rets, lw_turn),
    ]:
        m = calculate_portfolio_metrics(rets, benchmark_returns=ew_rets)
        m["turnover_medio"] = turn
        out[name] = m
        print(
            f"{name:14s}  Sharpe={m['sharpe_ratio']:+.3f}  "
            f"Ret={m['retorno_anualizado']*100:+.2f}%  Vol={m['volatilidad_anualizada']*100:.2f}%  "
            f"MDD={m['max_drawdown']*100:.2f}%  Turn={turn:.3f}"
        )

    with open("markowitz_lw_metrics.json", "w") as f:
        json.dump(out, f, indent=4)
    print("\nGuardado en markowitz_lw_metrics.json")
    return out


if __name__ == "__main__":
    run()
