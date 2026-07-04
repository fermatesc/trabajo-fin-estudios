"""
Análisis y veredicto estadístico de los resultados del backtest_harness.

Lee results_harness.json y, para cada configuración, responde a la pregunta central:
¿la variante cuántica supera (a) a su gemelo clásico de misma dimensión latente y
(b) al Equal-Weight, y con qué significancia? Usa evaluation/stat_significance.py
(Sharpe deflactado, test de diferencia de Sharpe Jobson-Korkie-Memmel, bootstrap de bloques).

Uso:
    python evaluation/analyze_harness.py --in results_harness.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.stat_significance import (annualized_sharpe, deflated_sharpe_ratio,
                                          sharpe_difference_test, block_bootstrap_ci)


def analyze(results, n_trials=None):
    """Analiza los resultados del harness y emite un veredicto por configuración.

    Para cada configuración con brazo cuántico, compara el Sharpe anualizado
    de la variante cuántica (Q) frente a su gemelo clásico (C) y frente a
    Equal-Weight (EW), con test de diferencia de Sharpe, Sharpe deflactado
    (DSR) y bootstrap de bloques.

    Args:
        results: dict cargado de results_harness.json (una entrada por config,
            con claves "quantum", "classical", "equal_weight").
        n_trials: nº de ensayos usado en el Sharpe deflactado. Si es None, se
            usa el nº de configuraciones cuánticas presentes en `results`.

    Returns:
        dict con un informe por configuración: Sharpes, dispersión entre
        semillas, tests Q_vs_C/Q_vs_EW, DSR, intervalos de confianza y un
        veredicto textual.

    Notes:
        El veredicto es "VENTAJA" solo si Q bate al gemelo de forma
        significativa (p<0.05) Y además supera a EW; si solo bate al gemelo
        de forma significativa pero no a EW, se marca aparte; si no supera al
        gemelo, se etiqueta como confirmación de la tesis NISQ-negativa.
        `ci_sharpe_diff_QmC` es el IC bootstrap de la diferencia diaria Q-C
        (mismo nº de días en ambas series); si no cruza 0 se considera robusto.
    """
    quantum_keys = [k for k in results if "quantum" in results[k]]
    if n_trials is None:
        n_trials = max(1, len(quantum_keys))

    report = {}
    rows = []
    for key in quantum_keys:
        r = results[key]
        q = np.array(r["quantum"]["ensemble_daily_returns"])
        c = np.array(r["classical"]["ensemble_daily_returns"])
        ew = np.array(r["equal_weight"]["daily_returns"])

        sr_q = annualized_sharpe(q)
        sr_c = annualized_sharpe(c)
        sr_ew = annualized_sharpe(ew)

        qc = sharpe_difference_test(q, c)
        qe = sharpe_difference_test(q, ew)
        dsr = deflated_sharpe_ratio(q, n_trials=n_trials)
        ci_q = block_bootstrap_ci(q, annualized_sharpe, n_boot=2000, block_size=21, seed=0)
        diff_daily = q - c
        ci_diff = block_bootstrap_ci(diff_daily,
                                     lambda x: annualized_sharpe(x, rf_annual=0.0),
                                     n_boot=2000, block_size=21, seed=0)

        beats_c = sr_q > sr_c
        beats_ew = sr_q > sr_ew
        sig_vs_c = (qc["p_value"] < 0.05) if not np.isnan(qc["p_value"]) else False
        sig_vs_ew = (qe["p_value"] < 0.05) if not np.isnan(qe["p_value"]) else False

        if beats_c and beats_ew and sig_vs_c:
            verdict = "VENTAJA: Q > gemelo (signif.) y > EW"
        elif beats_c and sig_vs_c:
            verdict = "Q > gemelo (signif.) pero NO bate EW"
        elif beats_c:
            verdict = "Q > gemelo (no signif.)"
        else:
            verdict = "Q NO supera al gemelo (tesis NISQ-negativa)"

        report[key] = {
            "config": r.get("config", {}),
            "sharpe": {"Q": sr_q, "C": sr_c, "EW": sr_ew},
            "sharpe_seed": {
                "Q_mean": r["quantum"]["metrics"].get("sharpe_seed_mean"),
                "Q_std": r["quantum"]["metrics"].get("sharpe_seed_std"),
                "C_mean": r["classical"]["metrics"].get("sharpe_seed_mean"),
                "C_std": r["classical"]["metrics"].get("sharpe_seed_std"),
            },
            "Q_vs_C": qc, "Q_vs_EW": qe,
            "deflated_sharpe_Q": dsr,
            "ci_sharpe_Q": ci_q,
            "ci_sharpe_diff_QmC": ci_diff,
            "verdict": verdict,
        }
        rows.append((key, sr_q, sr_c, sr_ew, qc["p_value"], dsr["dsr"],
                     ci_diff["ci_low"], ci_diff["ci_high"], verdict))

    print("\n" + "=" * 110)
    print(f"{'cfg':<6}{'SR_Q':>8}{'SR_C':>8}{'SR_EW':>8}{'p(Q-C)':>9}{'DSR_Q':>8}"
          f"{'ciQ-C_lo':>10}{'ciQ-C_hi':>10}  veredicto")
    print("-" * 110)
    for k, sq, sc, se, p, dsr, lo, hi, v in rows:
        def f(x):
            return f"{x:>8.3f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else f"{'nan':>8}"
        print(f"{k:<6}{f(sq)}{f(sc)}{f(se)}{p:>9.3f}{dsr:>8.3f}{lo:>10.3f}{hi:>10.3f}  {v}")
    print("=" * 110)
    print(f"(n_trials para Sharpe deflactado = {n_trials}; IC de la diferencia Q-C que NO cruza 0 => robusto)")
    return report


def main():
    """CLI: carga results_harness.json, ejecuta `analyze` e imprime/guarda el informe."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results_harness.json")
    ap.add_argument("--out", default="results_harness_analysis.json")
    ap.add_argument("--n-trials", type=int, default=0)
    args = ap.parse_args()

    with open(args.inp) as f:
        results = json.load(f)
    rep = analyze(results, n_trials=(args.n_trials or None))
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nAnálisis guardado en {args.out}")


if __name__ == "__main__":
    main()
