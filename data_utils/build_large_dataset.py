"""
Genera un dataset ESCALADO (~60 activos) en un directorio separado (data_large/)
para no sobrescribir los datos de 20 activos del cap. 5 de la tesis.

Solo genera los ficheros que consume data_utils/dataset_loader.py:
  prices_adjusted.csv, returns_log.csv, volatility_rolling.csv,
  rsi.csv, macd.csv, bb.csv, dynamic_correlations.npy
(Se omiten distancias euclidianas y sectores Wikipedia: el loader no los usa
 y son los pasos más lentos del pipeline original.)

TICKERS_LARGE contiene ~60 activos S&P 500 con histórico completo
2019-2024, diversificados por sector (Tech, Finanzas, Salud, Energía,
Consumo básico, Consumo discrecional, Industriales, Comunicaciones/Utilities).
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_pipeline import FinancialDataPipeline

TICKERS_LARGE = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'CSCO', 'ACN', 'AMD',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'AXP', 'BLK', 'SCHW',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX',
    'PG', 'WMT', 'KO', 'PEP', 'COST', 'MDLZ', 'CL',
    'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT',
    'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT',
    'DIS', 'VZ', 'T', 'CMCSA', 'NEE',
]

START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
WINDOW = 60
OUT_DIR = "data_large"


def main():
    """Ejecuta el pipeline escalado y guarda los resultados en `OUT_DIR`.

    Notes:
        Genera únicamente los ficheros que consume
        `data_utils/dataset_loader.py`: prices_adjusted.csv,
        returns_log.csv, volatility_rolling.csv, rsi.csv, macd.csv,
        bb.csv y dynamic_correlations.npy. Se omiten distancias
        euclidianas y sectores de Wikipedia porque el loader no los usa
        y son los pasos más lentos del pipeline original.
    """
    print(f"Construyendo dataset escalado: {len(TICKERS_LARGE)} activos -> {OUT_DIR}/", flush=True)
    p = FinancialDataPipeline(data_dir=OUT_DIR)

    prices = p.download_data(TICKERS_LARGE, START_DATE, END_DATE)
    print(f"Precios descargados: {prices.shape[1]} activos sobrevivieron al dropna, {prices.shape[0]} dias", flush=True)
    prices.to_csv(os.path.join(OUT_DIR, "prices_adjusted.csv"))

    returns = p.calculate_returns(prices)
    returns.to_csv(os.path.join(OUT_DIR, "returns_log.csv"))

    volatility = p.calculate_rolling_volatility(returns, WINDOW)
    volatility.to_csv(os.path.join(OUT_DIR, "volatility_rolling.csv"))

    rsi, macd, bb = p.calculate_technical_indicators(prices)
    rsi.to_csv(os.path.join(OUT_DIR, "rsi.csv"))
    macd.to_csv(os.path.join(OUT_DIR, "macd.csv"))
    bb.to_csv(os.path.join(OUT_DIR, "bb.csv"))

    corr = p.calculate_rolling_correlation(returns, WINDOW)
    np.save(os.path.join(OUT_DIR, "dynamic_correlations.npy"), corr)
    print(f"Correlaciones dinamicas: {corr.shape}", flush=True)

    print(f"\nDataset escalado listo en '{OUT_DIR}/'. Activos finales: {prices.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
