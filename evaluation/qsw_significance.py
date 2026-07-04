"""
Significancia estadística de la Caminata Estocástica Cuántica (QSW) frente al
Equal-Weight, sobre las series de retornos diarios almacenadas en
``results_qsw.json``.

Aplica a la QSW el MISMO escrutinio que el arnés riguroso aplica a las
variantes VQC (Sección "Programa Experimental Riguroso" de la memoria):

  1. Test de diferencia de Sharpe de Jobson-Korkie-Memmel (QSW vs Equal-Weight).
  2. Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014), con sensibilidad al
     numero de configuraciones ensayadas (multiple-testing).

Motivacion: la QSW es el unico asignador de la memoria cuyo Sharpe en punto
estimado supera al del Equal-Weight (1,37 vs 0,70). Este script comprueba si esa
ventaja sobrevive al contraste estadistico formal. Resultado: NO alcanza
significancia (p de JKM ~ 0,31) y el DSR cae por debajo del umbral de confianza
convencional en cuanto se corrige por multiplicidad.

Uso:
    python -m evaluation.qsw_significance
Salida:
    results_qsw_significance.json
"""
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.stat_significance import (
    annualized_sharpe,
    deflated_sharpe_ratio,
    sharpe_difference_test,
)

N_TRIALS_GRID = [1, 2, 6, 13]


def run(path="results_qsw.json", tau_key="tau_0.5"):
    """Calcula y guarda la significancia estadística de la QSW frente al Equal-Weight.

    Args:
        path: Ruta al fichero JSON con las series de retornos diarios de QSW y
            Equal-Weight, indexadas por umbral tau.
        tau_key: Clave del umbral tau a evaluar dentro del JSON de entrada.

    Returns:
        Diccionario con Sharpe anualizados, test de diferencia JKM/Memmel y
        Deflated Sharpe Ratio de QSW para la rejilla ``N_TRIALS_GRID``. El
        mismo diccionario se vuelca en ``results_qsw_significance.json``.

    Notes:
        ``N_TRIALS_GRID`` recoge los distintos supuestos de multiplicidad para
        el DSR: 1 = PSR clásico (sin corrección por selección), 2 = barrido
        del umbral tau (0.5, 0.3), 6 = tau x ablación del parámetro omega
        (4 valores), 13 = en línea con la multiplicidad del arnés VQC
        (13 configuraciones).
    """
    with open(path) as f:
        d = json.load(f)[tau_key]

    qsw = np.asarray(d["qsw"]["daily_returns"], dtype=float)
    ew = np.asarray(d["equal_weight"]["daily_returns"], dtype=float)

    out = {
        "tau_key": tau_key,
        "T": int(qsw.size),
        "sharpe_qsw_annual": annualized_sharpe(qsw),
        "sharpe_ew_annual": annualized_sharpe(ew),
        "jkm_qsw_vs_ew": sharpe_difference_test(qsw, ew),
        "deflated_sharpe_qsw": {
            str(n): deflated_sharpe_ratio(qsw, n_trials=n) for n in N_TRIALS_GRID
        },
    }

    print(f"T = {out['T']} jornadas")
    print(f"Sharpe QSW (anual, rf=4.5%): {out['sharpe_qsw_annual']:.3f}")
    print(f"Sharpe EW  (anual, rf=4.5%): {out['sharpe_ew_annual']:.3f}")
    jkm = out["jkm_qsw_vs_ew"]
    print(
        f"\nJKM/Memmel  diff={jkm['diff']:.3f}  z={jkm['z_stat']:.3f}  "
        f"p={jkm['p_value']:.4f}  -> {'significativo' if jkm['p_value'] < 0.05 else 'NO significativo'}"
    )
    print("\nDeflated Sharpe (QSW):")
    for n, ds in out["deflated_sharpe_qsw"].items():
        print(f"  n_trials={n:>2}:  DSR={ds['dsr']:.4f}")

    with open("results_qsw_significance.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nGuardado en results_qsw_significance.json")
    return out


if __name__ == "__main__":
    run()
