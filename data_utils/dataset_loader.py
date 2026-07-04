"""Dataset de PyTorch Geometric para grafos dinámicos financieros.

Transforma las series temporales generadas por `data_pipeline` (precios,
retornos, volatilidad, indicadores técnicos y correlaciones móviles) en una
secuencia de grafos `Data`, uno por instante temporal, listos para modelos
de PyTorch Geometric.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset


class FinancialGraphDataset(Dataset):
    """Dataset de PyTorch Geometric para transformar series temporales en grafos dinámicos.

    Attributes:
        data_dir: Directorio de datos de entrada.
        window: Tamaño de ventana usado para normalizar precios.
        tau: Umbral de correlación absoluta para crear aristas.
        assets: Lista de símbolos de los activos.
        num_assets: Número de activos.
        valid_dates: Fechas con datos válidos tras alinear volatilidad y correlaciones.
        x: Tensor de features de nodo, forma ``(T_valid, num_assets, 6)``.
    """

    def __init__(self, data_dir="data", window=60, tau=0.5):
        """Carga los datos preprocesados y construye las features de nodo.

        Args:
            data_dir: Directorio donde `data_pipeline` guardó los ficheros.
            window: Ventana usada para normalizar precios de forma causal.
            tau: Umbral de correlación absoluta para filtrar aristas.

        Notes:
            Las fechas válidas se alinean a partir de la volatilidad (que
            empieza tras `window` días); se valida que no haya más fechas
            válidas que matrices de correlación dinámica disponibles.
        """
        super().__init__()
        self.data_dir = data_dir
        self.window = window
        self.tau = tau

        self.prices = pd.read_csv(os.path.join(data_dir, "prices_adjusted.csv"), index_col=0, parse_dates=True)
        self.returns = pd.read_csv(os.path.join(data_dir, "returns_log.csv"), index_col=0, parse_dates=True)
        self.volatility = pd.read_csv(os.path.join(data_dir, "volatility_rolling.csv"), index_col=0, parse_dates=True)

        self.rsi = pd.read_csv(os.path.join(data_dir, "rsi.csv"), index_col=0, parse_dates=True)
        self.macd = pd.read_csv(os.path.join(data_dir, "macd.csv"), index_col=0, parse_dates=True)
        self.bb = pd.read_csv(os.path.join(data_dir, "bb.csv"), index_col=0, parse_dates=True)

        self.dynamic_corr = np.load(os.path.join(data_dir, "dynamic_correlations.npy"))

        self.assets = self.returns.columns.tolist()
        self.num_assets = len(self.assets)

        self.valid_dates = self.volatility.dropna().index

        assert len(self.valid_dates) <= len(self.dynamic_corr), \
            f"Desalineación: {len(self.valid_dates)} fechas válidas vs {len(self.dynamic_corr)} matrices de correlación"

        self.x = self._build_node_features()

    def _build_node_features(self):
        """Construye el tensor de features de nodo para todas las fechas válidas.

        Returns:
            Tensor de forma ``(T_valid, num_assets, 6)`` con las features
            precio normalizado, retorno, volatilidad, RSI, MACD y BB.

        Notes:
            Los precios se normalizan con media y desviación estándar
            móviles causales (ventana `self.window`) para evitar look-ahead
            bias. El RSI se imputa a 50 (neutral) y se escala a [0, 1]; el
            MACD se convierte a PPO (Percentage Price Oscillator) dividiendo
            entre el precio; volatilidad y retornos se escalan x100 para
            quedar en magnitud porcentual.
        """
        rets = self.returns.loc[self.valid_dates].values
        vols = self.volatility.loc[self.valid_dates].values

        rolling_mean = self.prices.rolling(window=self.window).mean()
        rolling_std = self.prices.rolling(window=self.window).std()
        prices_norm_df = (self.prices - rolling_mean) / (rolling_std + 1e-8)
        prices_norm = prices_norm_df.loc[self.valid_dates].values

        rsi = self.rsi.loc[self.valid_dates].values
        macd = self.macd.loc[self.valid_dates].values
        bb = self.bb.loc[self.valid_dates].values
        raw_prices = self.prices.loc[self.valid_dates].values

        rsi = np.nan_to_num(rsi, nan=50.0) / 100.0
        macd = np.nan_to_num(macd / (raw_prices + 1e-8))
        bb = np.nan_to_num(bb)
        prices_norm = np.nan_to_num(prices_norm)

        vols = vols * 100.0
        rets = rets * 100.0

        features = np.stack([prices_norm, rets, vols, rsi, macd, bb], axis=-1)
        return torch.tensor(features, dtype=torch.float32)

    def len(self):
        """Devuelve el número de instantes temporales (grafos) del dataset."""
        return len(self.valid_dates)

    def get(self, idx):
        """Genera el grafo G(t) en el instante `idx`.

        Args:
            idx: Índice temporal dentro de `self.valid_dates`.

        Returns:
            Objeto `Data` de PyTorch Geometric con `x` (features de nodo),
            `edge_index` y `edge_attr` (topología basada en correlación),
            `y` (retornos del día siguiente) y `cov` (matriz de covarianza).

        Notes:
            Las aristas se generan aplicando el umbral `tau` sobre el valor
            absoluto de la correlación, manteniendo explícitamente los
            auto-bucles de la diagonal. La covarianza se reconstruye como
            `Cov(i,j) = Corr(i,j) * Vol(i) * Vol(j)` mediante el producto
            externo de las volatilidades.
        """
        x = self.x[idx]

        if idx < self.len() - 1:
            next_date = self.valid_dates[idx + 1]
            y = torch.tensor(self.returns.loc[next_date].values, dtype=torch.float32)
        else:
            y = torch.zeros(self.num_assets, dtype=torch.float32)

        corr_matrix = self.dynamic_corr[idx]

        mask = (np.abs(corr_matrix) > self.tau)
        indices = np.where(mask)

        if len(indices[0]) > 0:
            edge_index = torch.tensor(np.stack(indices), dtype=torch.long)
            edge_attr = torch.tensor(np.abs(corr_matrix[indices]), dtype=torch.float32).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        vol_t = self.volatility.loc[self.valid_dates[idx]].values
        vol_outer = np.outer(vol_t, vol_t)
        cov_matrix = corr_matrix * vol_outer
        cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cov=cov_tensor)


if __name__ == "__main__":
    dataset = FinancialGraphDataset(tau=0.5)
    print(f"Dataset listo. Total de ventanas temporales (grafos): {len(dataset)}")

    sample_data = dataset[0]
    print(f"\nInfo del Grafo T=0:")
    print(f"- Nodos (Acciones): {sample_data.num_nodes}")
    print(f"- Atributos por nodo: {sample_data.num_node_features} (Precio, Retorno, Volatilidad, RSI, MACD, BB)")
    print(f"- Aristas (Conexiones activas): {sample_data.num_edges}")
