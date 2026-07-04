"""
Genera un dataset de HISTORIA EXTENDIDA (2005–2024) en un directorio separado (data_xl/)
para no sobrescribir los datos existentes en data/ ni data_large/.

Solo genera los ficheros que consume data_utils/dataset_loader.py:
  prices_adjusted.csv, returns_log.csv, volatility_rolling.csv,
  rsi.csv, macd.csv, bb.csv, dynamic_correlations.npy
(Se omiten distancias euclidianas y sectores Wikipedia: el loader no los usa
 y son los pasos más lentos del pipeline original.)

Los tickers sin histórico completo desde 2005 se eliminan automáticamente
por el dropna(axis=1) dentro de FinancialDataPipeline.download_data().

TICKERS_XL contiene ~62 tickers S&P 500 diversificados por sector (Tech,
Finanzas, Salud, Energía, Consumo básico, Consumo discrecional,
Industriales, Comunicaciones/Utilities), la misma lista que usa
build_large_dataset.py.
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_pipeline import FinancialDataPipeline

TICKERS_XL = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'CSCO', 'ACN', 'AMD',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'AXP', 'BLK', 'SCHW',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX',
    'PG', 'WMT', 'KO', 'PEP', 'COST', 'MDLZ', 'CL',
    'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT',
    'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT',
    'DIS', 'VZ', 'T', 'CMCSA', 'NEE',
]

START_DATE = "2005-01-01"
END_DATE = "2024-01-01"
WINDOW = 60
OUT_DIR = "data_xl"


def main():
    """Ejecuta el pipeline extendido y guarda los resultados en `OUT_DIR`.

    Notes:
        Genera únicamente los ficheros que consume
        `data_utils/dataset_loader.py`: prices_adjusted.csv,
        returns_log.csv, volatility_rolling.csv, rsi.csv, macd.csv,
        bb.csv y dynamic_correlations.npy. Se omiten distancias
        euclidianas y sectores de Wikipedia porque el loader no los usa
        y son los pasos más lentos del pipeline original. Los tickers sin
        histórico completo desde 2005 se eliminan automáticamente por el
        `dropna(axis=1)` dentro de `FinancialDataPipeline.download_data()`.
    """
    print(f"Construyendo dataset extendido 2005-2024: {len(TICKERS_XL)} activos candidatos -> {OUT_DIR}/", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    p = FinancialDataPipeline(data_dir=OUT_DIR)

    prices = p.download_data(TICKERS_XL, START_DATE, END_DATE)
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

    print(f"\nDataset extendido listo en '{OUT_DIR}/'. Activos finales: {prices.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
