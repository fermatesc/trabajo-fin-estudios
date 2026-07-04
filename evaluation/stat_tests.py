"""
Tests estadísticos formales sobre las series del proyecto (citados en el cap. 4 de la memoria):
- Jarque-Bera sobre retornos logarítmicos (normalidad).
- Augmented Dickey-Fuller sobre precios y sobre retornos (estacionariedad).
Genera data/stat_tests.json y un resumen por consola.
"""
import os
import sys
import json

import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


def main():
    """Calcula Jarque-Bera y ADF (sobre precios y retornos) para cada ticker
    y guarda los resultados en ``data/stat_tests.json``, imprimiendo un
    resumen agregado por consola.

    Returns:
        None.
    """
    prices = pd.read_csv(os.path.join(DATA_DIR, "prices_adjusted.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(DATA_DIR, "returns_log.csv"), index_col=0, parse_dates=True)

    results = {}
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        p = prices[ticker].dropna()

        jb_stat, jb_p = stats.jarque_bera(r)
        adf_ret = adfuller(r, autolag="AIC")
        adf_price = adfuller(p, autolag="AIC")

        results[ticker] = {
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_p),
            "adf_returns_stat": float(adf_ret[0]),
            "adf_returns_pvalue": float(adf_ret[1]),
            "adf_prices_stat": float(adf_price[0]),
            "adf_prices_pvalue": float(adf_price[1]),
            "kurtosis_exceso": float(stats.kurtosis(r)),
            "skewness": float(stats.skew(r)),
        }

    out_path = os.path.join(DATA_DIR, "stat_tests.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n = len(results)
    jb_reject = sum(1 for v in results.values() if v["jarque_bera_pvalue"] < 0.01)
    adf_ret_reject = sum(1 for v in results.values() if v["adf_returns_pvalue"] < 0.01)
    adf_price_reject = sum(1 for v in results.values() if v["adf_prices_pvalue"] < 0.05)
    jb_min = min(v["jarque_bera_stat"] for v in results.values())
    kurt_med = sorted(v["kurtosis_exceso"] for v in results.values())[n // 2]

    print(f"Resultados guardados en {out_path}")
    print(f"Jarque-Bera: se rechaza normalidad (p<0.01) en {jb_reject}/{n} series; JB minimo = {jb_min:.1f}")
    print(f"Curtosis en exceso mediana: {kurt_med:.2f}")
    print(f"ADF retornos: estacionarios (p<0.01) en {adf_ret_reject}/{n} series")
    print(f"ADF precios: se rechaza raiz unitaria (p<0.05) en {adf_price_reject}/{n} series (esperado: ~0)")


if __name__ == "__main__":
    main()
