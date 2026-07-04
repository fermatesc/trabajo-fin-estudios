"""Pipeline de descarga y preprocesamiento de series temporales financieras.

Descarga precios ajustados con ``yfinance``, calcula retornos, volatilidad,
indicadores técnicos y varias matrices de adyacencia (correlación, distancia
euclidiana, sectores GICS) para la construcción de grafos dinámicos y
estáticos usados por los modelos de la cartera.
"""
import os
from typing import List, Optional
import pandas as pd
import yfinance as yf
import numpy as np


class FinancialDataPipeline:
    """Pipeline para la descarga y preprocesamiento de series temporales financieras.

    Prepara los datos para la construcción de grafos dinámicos.

    Attributes:
        data_dir: Ruta del directorio donde se guardan los ficheros generados.
    """

    def __init__(self, data_dir: str = "data"):
        """Inicializa el pipeline y crea el directorio de datos si no existe.

        Args:
            data_dir: Ruta del directorio donde se guardarán los datos.
        """
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Descarga datos históricos de precios de cierre ajustados.

        Args:
            tickers: Lista de símbolos bursátiles a descargar.
            start_date: Fecha de inicio en formato ``YYYY-MM-DD``.
            end_date: Fecha de fin en formato ``YYYY-MM-DD``.

        Returns:
            DataFrame de precios de cierre, con festivos rellenados hacia
            adelante y activos sin histórico completo descartados.

        Raises:
            ValueError: Si `tickers` está vacía.

        Notes:
            Se usa `ffill` para festivos y `dropna` para descartar activos
            sin histórico completo, evitando así look-ahead bias.
        """
        if not tickers:
            raise ValueError("La lista de tickers no puede estar vacía.")

        print(f"Descargando datos para {len(tickers)} activos desde {start_date} hasta {end_date}...")

        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']

        data = data.ffill().dropna(axis=1)
        return data

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calcula los retornos logarítmicos diarios (``ln(P_t / P_{t-1})``).

        Args:
            prices: DataFrame de precios de cierre.

        Returns:
            DataFrame de retornos logarítmicos, sin la primera fila (NaN).

        Notes:
            Los retornos logarítmicos son preferibles para series temporales
            financieras por su aditividad.
        """
        print("Calculando retornos logarítmicos...")
        returns = np.log(prices / prices.shift(1))
        return returns.dropna()

    def calculate_rolling_volatility(self, returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """Calcula la volatilidad móvil de los retornos logarítmicos.

        Se usa como feature en los nodos del grafo dinámico.

        Args:
            returns: DataFrame de retornos logarítmicos.
            window: Tamaño de la ventana móvil en días.

        Returns:
            DataFrame con la desviación estándar móvil, sin los periodos
            iniciales incompletos.
        """
        print(f"Calculando volatilidad móvil con ventana de {window} días...")
        rolling_vol = returns.rolling(window=window).std()
        return rolling_vol.dropna()

    def calculate_rolling_correlation(self, returns: pd.DataFrame, window: int = 60) -> np.ndarray:
        """Calcula la matriz de correlación móvil (Pearson) para grafos dinámicos.

        Args:
            returns: DataFrame de retornos logarítmicos.
            window: Tamaño de la ventana móvil en días.

        Returns:
            Array 3D de forma ``(tiempo, num_activos, num_activos)`` con las
            matrices de correlación en cada instante.
        """
        print(f"Calculando matrices de correlación con ventana de {window} días...")
        rolling_corr = returns.rolling(window=window).corr()

        rolling_corr = rolling_corr.dropna()

        num_assets = returns.shape[1]
        num_time_steps = len(rolling_corr) // num_assets

        corr_matrices = rolling_corr.values.reshape(num_time_steps, num_assets, num_assets)
        return corr_matrices

    def calculate_wikipedia_sector_edges(self, tickers: List[str]) -> np.ndarray:
        """Construye una matriz de adyacencia estática a partir de sectores GICS.

        Descarga la lista del S&P 500 desde Wikipedia y marca con 1 los pares
        de activos que comparten sector GICS (0 en caso contrario).

        Args:
            tickers: Lista de símbolos bursátiles.

        Returns:
            Matriz de adyacencia ``(num_activos, num_activos)``. Si falla la
            descarga, se devuelve la matriz identidad como respaldo.

        Notes:
            yfinance usa '-' en los tickers (p. ej. BRK-B) mientras que
            Wikipedia usa '.' (p. ej. BRK.B), por lo que se normalizan antes
            de buscar el sector. Se usa `StringIO` con `pd.read_html` para
            evitar warnings de pandas al pasar el texto de la respuesta.
        """
        print("Obteniendo sectores de Wikipedia (GICS) para grafo estático...")
        try:
            import requests
            from io import StringIO
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            tables = pd.read_html(StringIO(response.text))
            sp500_df = tables[0]

            sector_map = dict(zip(sp500_df['Symbol'], sp500_df['GICS Sector']))

            num_assets = len(tickers)
            adj_matrix = np.zeros((num_assets, num_assets))

            for i in range(num_assets):
                for j in range(num_assets):
                    if i == j:
                        adj_matrix[i, j] = 1.0
                    else:
                        t_i = tickers[i].replace('-', '.')
                        t_j = tickers[j].replace('-', '.')
                        sector_i = sector_map.get(t_i, f"Unknown_{i}")
                        sector_j = sector_map.get(t_j, f"Unknown_{j}")

                        if sector_i == sector_j and not sector_i.startswith("Unknown"):
                            adj_matrix[i, j] = 1.0
            return adj_matrix
        except Exception as e:
            print(f"Error al obtener datos de Wikipedia: {e}. Se usará matriz identidad.")
            return np.eye(len(tickers))

    def calculate_distance_edges(self, returns: pd.DataFrame, window: int = 60) -> np.ndarray:
        """Calcula matrices de similitud dinámicas basadas en distancia euclidiana.

        Args:
            returns: DataFrame de retornos logarítmicos.
            window: Tamaño de la ventana temporal en días.

        Returns:
            Array 3D de forma ``(tiempo, num_activos, num_activos)`` con la
            inversa de la distancia euclidiana entre activos en cada ventana,
            o un array vacío si no hay suficientes datos.
        """
        print(f"Calculando matrices de distancia euclidiana con ventana de {window} días...")
        num_assets = returns.shape[1]
        num_time_steps = len(returns) - window + 1

        if num_time_steps <= 0:
            return np.array([])

        dist_matrices = np.zeros((num_time_steps, num_assets, num_assets))

        for t in range(num_time_steps):
            window_data = returns.iloc[t:t+window].values
            for i in range(num_assets):
                for j in range(num_assets):
                    if i == j:
                        dist_matrices[t, i, j] = 1.0
                    else:
                        dist = np.linalg.norm(window_data[:, i] - window_data[:, j])
                        dist_matrices[t, i, j] = 1.0 / (1.0 + dist)

        return dist_matrices

    def calculate_technical_indicators(self, prices: pd.DataFrame) -> tuple:
        """Calcula RSI, MACD y Bandas de Bollinger.

        Args:
            prices: DataFrame de precios de cierre.

        Returns:
            Tupla ``(rsi, macd, bb)`` con el RSI de 14 días, el MACD (12, 26)
            y el %B de las Bandas de Bollinger a 20 días.
        """
        print("Calculando RSI, MACD y Bandas de Bollinger...")
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2

        rolling_mean = prices.rolling(window=20).mean()
        rolling_std = prices.rolling(window=20).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        bb = (prices - lower_band) / (upper_band - lower_band + 1e-8)

        return rsi, macd, bb

    def run_pipeline(self, tickers: List[str], start_date: str, end_date: str, window: int = 60):
        """Ejecuta el pipeline completo y guarda los resultados en `self.data_dir`.

        Args:
            tickers: Lista de símbolos bursátiles.
            start_date: Fecha de inicio en formato ``YYYY-MM-DD``.
            end_date: Fecha de fin en formato ``YYYY-MM-DD``.
            window: Tamaño de ventana en días para los cálculos móviles.

        Raises:
            Exception: Relanza cualquier excepción ocurrida durante el pipeline.

        Notes:
            Genera los ficheros: prices_adjusted.csv, returns_log.csv,
            volatility_rolling.csv, rsi.csv, macd.csv, bb.csv,
            dynamic_correlations.npy, static_wikipedia_sectors.npy y,
            si hay datos suficientes, dynamic_distances.npy.
        """
        try:
            prices = self.download_data(tickers, start_date, end_date)
            prices_path = os.path.join(self.data_dir, "prices_adjusted.csv")
            prices.to_csv(prices_path)

            returns = self.calculate_returns(prices)
            returns_path = os.path.join(self.data_dir, "returns_log.csv")
            returns.to_csv(returns_path)

            volatility = self.calculate_rolling_volatility(returns, window)
            volatility_path = os.path.join(self.data_dir, "volatility_rolling.csv")
            volatility.to_csv(volatility_path)

            rsi, macd, bb = self.calculate_technical_indicators(prices)
            rsi.to_csv(os.path.join(self.data_dir, "rsi.csv"))
            macd.to_csv(os.path.join(self.data_dir, "macd.csv"))
            bb.to_csv(os.path.join(self.data_dir, "bb.csv"))

            corr_matrices = self.calculate_rolling_correlation(returns, window)
            corr_path = os.path.join(self.data_dir, "dynamic_correlations.npy")
            np.save(corr_path, corr_matrices)

            wiki_adj_matrix = self.calculate_wikipedia_sector_edges(tickers)
            wiki_path = os.path.join(self.data_dir, "static_wikipedia_sectors.npy")
            np.save(wiki_path, wiki_adj_matrix)

            dist_matrices = self.calculate_distance_edges(returns, window)
            if dist_matrices.size > 0:
                dist_path = os.path.join(self.data_dir, "dynamic_distances.npy")
                np.save(dist_path, dist_matrices)

            print(f"Pipeline completado. Datos guardados en '{self.data_dir}'.")

        except Exception as e:
            print(f"Error en el pipeline: {str(e)}")
            raise e


if __name__ == "__main__":
    sp500_sample = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA',
        'JPM', 'BAC', 'GS', 'MS',
        'JNJ', 'PFE', 'UNH', 'ABBV',
        'XOM', 'CVX', 'COP', 'SLB',
        'PG', 'WMT', 'KO', 'PEP'
    ]

    pipeline = FinancialDataPipeline(data_dir="data")
    pipeline.run_pipeline(
        tickers=sp500_sample,
        start_date="2019-01-01",
        end_date="2024-01-01",
        window=60
    )
