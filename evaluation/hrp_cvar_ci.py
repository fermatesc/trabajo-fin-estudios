"""
Intervalo de confianza para la REDUCCION de CVaR-95 de HRP frente al
Equal-Weight, sobre la historia extendida (2005-2024, data_xl) con rebalanceo
mensual (freq=21) y costes netos.

La memoria afirma que HRP reduce el CVaR-95 "del orden del 16-17%" frente al
Equal-Weight, pero esa cifra era un punto estimado sin medida de incertidumbre.
Este script:
  1. Reproduce las series de retornos diarios NETOS de Equal-Weight y HRP con el
     mismo backtest realista que `training/evaluate_rebalance.py`
     (`backtest_with_freq`, deriva de pesos, coste solo el dia de rebalanceo).
  2. Calcula la reduccion relativa de CVaR-95 (HRP vs EW).
  3. Estima un IC al 95% por bootstrap estacionario de bloques PAREADO
     (remuestrea los mismos indices de bloque para ambas series, preservando la
     sincronia temporal EW<->HRP), reportando el IC de la reduccion absoluta y
     relativa del CVaR-95.

Uso:
    .venv\\Scripts\\python.exe -m evaluation.hrp_cvar_ci
Salida:
    results_hrp_cvar_ci.json
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.sequence_dataset import SequenceGraphDataset
from training.evaluate_rebalance import (
    backtest_with_freq,
    make_ew_fn,
    make_hrp_fn,
)
from training.evaluate_risk import build_splits, historical_cvar95

DATASET = "data_xl"
FREQ = 21
N_BOOT = 5000
BLOCK = 21
SEED = 0


def paired_block_bootstrap_cvar(ew, hrp, n_boot=N_BOOT, block=BLOCK, seed=SEED):
    """Bootstrap estacionario PAREADO de la reduccion de CVaR-95 (HRP vs EW).

    Args:
        ew: array 1D de retornos diarios netos de Equal-Weight.
        hrp: array 1D de retornos diarios netos de HRP, misma longitud que ew.
        n_boot: número de réplicas bootstrap.
        block: longitud media esperada de los bloques (en periodos).
        seed: semilla para np.random.default_rng.

    Returns:
        Tupla (abs_red, rel_red, ci_abs, ci_rel):
            abs_red: array (n_boot,) con CVaR_EW - CVaR_HRP por réplica.
            rel_red: array (n_boot,) con (CVaR_EW - CVaR_HRP) / CVaR_EW por réplica.
            ci_abs: tupla (p2.5, p97.5) de abs_red.
            ci_rel: tupla (p2.5, p97.5) de rel_red.

    Notes:
        Ambas series se remuestrean con los MISMOS índices de bloque en cada
        réplica, preservando la sincronía temporal EW<->HRP (bootstrap pareado).
    """
    ew = np.asarray(ew, float)
    hrp = np.asarray(hrp, float)
    T = ew.size
    rng = np.random.default_rng(seed)
    p_geom = 1.0 / max(block, 1)

    abs_red = np.empty(n_boot)
    rel_red = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        filled = 0
        while filled < T:
            start = rng.integers(0, T)
            length = min(rng.geometric(p_geom), T - filled)
            idx[filled:filled + length] = (start + np.arange(length)) % T
            filled += length
        c_ew = historical_cvar95(ew[idx])
        c_hrp = historical_cvar95(hrp[idx])
        abs_red[b] = c_ew - c_hrp
        rel_red[b] = (c_ew - c_hrp) / c_ew if c_ew != 0 else np.nan

    def ci(a):
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return abs_red, rel_red, ci(abs_red), ci(rel_red)


def run():
    """Ejecuta el backtest EW/HRP, calcula la reducción de CVaR-95 y su IC
    bootstrap, e imprime y guarda el resultado en results_hrp_cvar_ci.json."""
    ds = SequenceGraphDataset(data_dir=DATASET, tau=0.5, lookback=20)
    n_assets = ds.num_assets
    _, test_indices, total, train_size, _ = build_splits(ds)
    print(f"{DATASET}: {n_assets} activos | test {len(test_indices)} jornadas | freq={FREQ}")

    ew_rets, _ = backtest_with_freq(make_ew_fn(n_assets), ds, test_indices, n_assets, FREQ)
    hrp_rets, _ = backtest_with_freq(make_hrp_fn(), ds, test_indices, n_assets, FREQ)

    cvar_ew = historical_cvar95(ew_rets)
    cvar_hrp = historical_cvar95(hrp_rets)
    abs_point = cvar_ew - cvar_hrp
    rel_point = abs_point / cvar_ew

    _, _, (abs_lo, abs_hi), (rel_lo, rel_hi) = paired_block_bootstrap_cvar(ew_rets, hrp_rets)

    out = {
        "dataset": DATASET, "freq": FREQ, "n_boot": N_BOOT, "block": BLOCK,
        "T": len(test_indices),
        "cvar95_ew": float(cvar_ew),
        "cvar95_hrp": float(cvar_hrp),
        "reduccion_absoluta": float(abs_point),
        "reduccion_absoluta_ci95": [abs_lo, abs_hi],
        "reduccion_relativa": float(rel_point),
        "reduccion_relativa_ci95": [rel_lo, rel_hi],
    }
    print(f"CVaR-95 EW  = {cvar_ew*100:.3f}%")
    print(f"CVaR-95 HRP = {cvar_hrp*100:.3f}%")
    print(f"Reduccion relativa = {rel_point*100:.1f}%  "
          f"IC95 = [{rel_lo*100:.1f}%, {rel_hi*100:.1f}%]")
    print(f"  -> {'IC excluye 0 (significativa)' if rel_lo > 0 else 'IC incluye 0'}")

    with open("results_hrp_cvar_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Guardado en results_hrp_cvar_ci.json")
    return out


if __name__ == "__main__":
    run()
