"""
Backtest walk-forward del asignador Quantum Stochastic Walk (QSW).

Evalúa el brazo QSW (models/qsw_portfolio.py) con el MISMO protocolo que el
harness riguroso (evaluation/backtest_harness.py): mismo split (test_start en el
70%), retornos simples netos de 10 pb de coste, drift de pesos entre rebalanceos,
y benchmark Equal-Weight idéntico. A diferencia de la QGNN, el QSW no entrena
parámetros cuánticos: en cada refit se calibran (omega, alpha) por búsqueda en
rejilla sobre la ventana in-sample (Sharpe máximo) y se aplican out-of-sample.

Se ejecuta una vez por umbral de grafo tau (0.5 y 0.3), de forma desacoplada de
la batería de pérdidas, y guarda retornos diarios para los tests de significancia.

Uso:
    python evaluation/qsw_backtest.py
    python evaluation/qsw_backtest.py --smoke
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRANSACTION_COST_BPS, TRAIN_RATIO
from data_utils.dataset_loader import FinancialGraphDataset
from models.qsw_portfolio import qsw_weights
from evaluation.metrics import calculate_portfolio_metrics

REFIT_FREQ = 21
MU_WINDOW = 60
OMEGA_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
ALPHA_GRID = [0.0, 1.0, 2.0, 3.0, 4.0]
CAL_WINDOW = 120
COST = TRANSACTION_COST_BPS / 10000.0


def expected_returns(returns_df, valid_dates, idx):
    """Retorno esperado causal: media de log-retornos en [idx-MU_WINDOW, idx)."""
    lo = max(0, idx - MU_WINDOW)
    window = returns_df.loc[valid_dates[lo:idx]]
    if len(window) == 0:
        return np.zeros(returns_df.shape[1])
    return window.mean().values


def calibrate(corr_stack, returns_arr, returns_df, valid_dates, cal_end, tau,
              fixed_omega=None):
    """
    Elige (omega, alpha) que maximizan el Sharpe in-sample. Heurística barata:
    para cada combinación se derivan los pesos con la correlación/retorno del
    último día in-sample y se puntúan sobre los retornos realizados de la ventana
    de calibración (sin tocar el periodo de test).

    Si fixed_omega no es None, omega se fija (ablación cuántico/clásico) y solo se
    calibra alpha.

    Notes:
        `realized`/`realized_simple` tienen shape (W, N): W días de la
        ventana de calibración (CAL_WINDOW, recortada si no hay suficiente
        historial) por N activos.
    """
    lo = max(0, cal_end - CAL_WINDOW)
    if cal_end - lo < 5:
        return (0.5 if fixed_omega is None else fixed_omega), 0.0
    realized = returns_arr[lo:cal_end]
    realized_simple = np.expm1(realized)
    C = corr_stack[cal_end - 1]
    mu = expected_returns(returns_df, valid_dates, cal_end - 1)
    omega_grid = OMEGA_GRID if fixed_omega is None else [fixed_omega]
    best, best_sharpe = (omega_grid[0], 0.0), -1e9
    for omega in omega_grid:
        for alpha in ALPHA_GRID:
            w = qsw_weights(C, omega=omega, mu=mu, alpha=alpha, tau=tau)
            port = realized_simple @ w
            sd = port.std()
            sharpe = port.mean() / sd if sd > 1e-9 else -1e9
            if sharpe > best_sharpe:
                best_sharpe, best = sharpe, (omega, alpha)
    return best


def run_qsw(tau, test_indices, dataset, smoke=False, fixed_omega=None):
    """Ejecuta el walk-forward del QSW sobre `test_indices` con refit periódico.

    Args:
        tau: umbral del grafo de correlación.
        test_indices: índices del dataset a recorrer en orden.
        dataset: dataset con dynamic_corr, returns y valid_dates.
        smoke: si True, no calibra (omega, alpha) y usa omega=0.5 fijo.
        fixed_omega: si no es None, fija omega para la ablación cuántico/clásico
            y solo se calibra alpha.

    Returns:
        Tupla (daily_simple_returns_net, mean_turnover, (omega, alpha)) con
        los últimos parámetros calibrados.

    Notes:
        Los pesos decididos al cierre del día i ganan el retorno del día
        i+1 (sin look-ahead), la misma convención que el harness
        (sample.y = retorno de idx+1).
    """
    corr_stack = dataset.dynamic_corr
    returns_df = dataset.returns
    valid_dates = dataset.valid_dates
    returns_arr = returns_df.loc[valid_dates].values
    N = dataset.num_assets

    w_prev = np.ones(N) / N
    omega = 0.5 if fixed_omega is None else fixed_omega
    alpha = 0.0
    rets, turns = [], []
    for k, i in enumerate(test_indices):
        if k % REFIT_FREQ == 0 and not smoke:
            omega, alpha = calibrate(corr_stack, returns_arr, returns_df,
                                     valid_dates, i, tau, fixed_omega=fixed_omega)
        if i + 1 >= len(returns_arr):
            break
        C = corr_stack[i]
        mu = expected_returns(returns_df, valid_dates, i)
        w = qsw_weights(C, omega=omega, mu=mu, alpha=alpha, tau=tau)
        y_simple = np.expm1(returns_arr[i + 1])
        turnover = np.abs(w - w_prev).sum()
        ret = float(np.dot(w, y_simple) - turnover * COST)
        rets.append(ret); turns.append(float(turnover))
        w_drift = w * (1.0 + y_simple)
        w_prev = w_drift / w_drift.sum()
    return np.array(rets), float(np.mean(turns)), (omega, alpha)


def equal_weight(dataset, test_indices):
    """Retornos diarios simples netos de costes de la cartera Equal-Weight,
    con la misma convención de timing que `run_qsw` (sin look-ahead)."""
    N = dataset.num_assets
    returns_arr = dataset.returns.loc[dataset.valid_dates].values
    w_t = np.ones(N) / N
    w_prev = w_t.copy()
    rets = []
    for i in test_indices:
        if i + 1 >= len(returns_arr):
            break
        y_simple = np.expm1(returns_arr[i + 1])
        turnover = np.abs(w_t - w_prev).sum()
        rets.append(float(np.dot(w_t, y_simple) - turnover * COST))
        w_drift = w_t * (1.0 + y_simple)
        w_prev = w_drift / w_drift.sum()
    return np.array(rets)


def main():
    """CLI: ejecuta el backtest QSW para cada tau pedido y acumula resultados
    en un único JSON de salida (--out)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="sin calibración, omega=0.5 fijo")
    ap.add_argument("--fixed-omega", type=float, default=None,
                    help="ablación: fija omega (0=cuántico puro, 1=clásico puro) y calibra solo alpha")
    ap.add_argument("--taus", default="0.5,0.3", help="lista de tau separada por comas")
    ap.add_argument("--data-dir", default="data",
                    help="directorio del dataset (p. ej. data_xl para la historia 2005-2024)")
    ap.add_argument("--train-ratio", type=float, default=None,
                    help="override de TRAIN_RATIO (p. ej. 0.25 para que el OOS cubra 2008 y 2020)")
    ap.add_argument("--out", default="results_qsw.json")
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    tr = TRAIN_RATIO if args.train_ratio is None else args.train_ratio
    out = {}
    for tau in taus:
        ds = FinancialGraphDataset(data_dir=args.data_dir, tau=tau)
        total = len(ds)
        test_start = int(total * tr)
        test_indices = list(range(test_start, total))
        tag = f"tau={tau}" + (f" fixed_omega={args.fixed_omega}" if args.fixed_omega is not None else "")
        print(f"\n=== QSW {tag}  total={total} test_start={test_start} "
              f"test_days={len(test_indices)} ===")
        ew_rets = equal_weight(ds, test_indices)
        qsw_rets, qsw_turn, last_params = run_qsw(tau, test_indices, ds,
                                                  smoke=args.smoke, fixed_omega=args.fixed_omega)
        qsw_m = calculate_portfolio_metrics(qsw_rets, benchmark_returns=ew_rets)
        qsw_m["turnover_medio"] = qsw_turn
        ew_m = calculate_portfolio_metrics(ew_rets, benchmark_returns=ew_rets)
        print(f"  QSW Sharpe={qsw_m['sharpe_ratio']:.3f} turn={qsw_turn:.3f} "
              f"(omega,alpha)_last={last_params}  |  EW Sharpe={ew_m['sharpe_ratio']:.3f}")
        out[f"tau_{tau}"] = {
            "tau": tau,
            "test_days": len(test_indices),
            "qsw": {"metrics": qsw_m, "daily_returns": qsw_rets.tolist(),
                    "last_params": list(last_params)},
            "equal_weight": {"metrics": ew_m, "daily_returns": ew_rets.tolist()},
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  guardado tau_{tau} en {args.out}")


if __name__ == "__main__":
    main()
