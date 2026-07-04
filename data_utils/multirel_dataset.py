"""
Variante multi-relacional de FinancialGraphDataset.

Enriquece el atributo de arista de un escalar (|corr|) a un vector de 3
relaciones por arista:

    edge_attr = [ |corr_ij| ,  sign(corr_ij) ,  mismo_sector_ij ]

con lo que el GAT dispone de magnitud, signo y pertenencia sectorial en lugar de
solo la magnitud de la correlación. El grafo (edge_index) es IDÉNTICO al del
dataset base (mismo umbral tau), de modo que el experimento aísla el efecto de
ENRIQUECER las aristas, no el de cambiar la topología.

Encuadre honesto: una arista más rica beneficia a la GNN —clásica y cuántica por
igual— y NO constituye por sí misma una ventaja cuántica. Sirve para comprobar si
una representación de arista más informativa mejora el Sharpe del bloque GNN.

Notas sobre los mapas de sectores del módulo:
    _SECTOR_IDS: sectores de los 20 tickers originales (4 por sector, según
        config.TICKERS), en el orden Tech, Finanzas, Salud, Energía, Consumo.
    _GICS: mapa GICS por ticker para universos extendidos (p. ej. data_xl,
        58 activos); solo se usa cuando el universo no es el de los 20
        originales, caso en el que se conserva _SECTOR_IDS byte a byte para
        no alterar los resultados publicados del cap. 5. Un ticker ausente
        del mapa cae en su propio sector singleton (su símbolo como
        categoría), lo que evita inducir aristas same-sector espurias.
"""
import os
import numpy as np
import torch
from torch_geometric.data import Data

from data_utils.dataset_loader import FinancialGraphDataset

_SECTOR_IDS = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
_ORIG_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "BAC", "GS", "MS",
                 "JNJ", "PFE", "UNH", "ABBV", "XOM", "CVX", "COP", "SLB",
                 "PG", "WMT", "KO", "PEP"]

_GICS = {
    "AAPL": "IT", "MSFT": "IT", "NVDA": "IT", "AMD": "IT", "ADBE": "IT",
    "CRM": "IT", "CSCO": "IT", "ORCL": "IT", "ACN": "IT",
    "GOOGL": "COMM", "CMCSA": "COMM", "DIS": "COMM", "T": "COMM", "VZ": "COMM",
    "JPM": "FIN", "BAC": "FIN", "GS": "FIN", "MS": "FIN", "C": "FIN",
    "WFC": "FIN", "AXP": "FIN", "BLK": "FIN", "SCHW": "FIN",
    "JNJ": "HLTH", "PFE": "HLTH", "UNH": "HLTH", "ABBV": "HLTH", "ABT": "HLTH",
    "AMGN": "HLTH", "BMY": "HLTH", "LLY": "HLTH", "MRK": "HLTH", "TMO": "HLTH",
    "XOM": "ENE", "CVX": "ENE", "COP": "ENE", "SLB": "ENE", "EOG": "ENE",
    "PG": "STPL", "WMT": "STPL", "KO": "STPL", "PEP": "STPL", "CL": "STPL",
    "COST": "STPL", "MDLZ": "STPL",
    "AMZN": "DISC", "HD": "DISC", "LOW": "DISC", "MCD": "DISC", "NKE": "DISC",
    "SBUX": "DISC", "TGT": "DISC",
    "BA": "IND", "CAT": "IND", "HON": "IND", "LMT": "IND", "RTX": "IND", "UPS": "IND",
    "NEE": "UTIL",
}

EDGE_DIM = 3


class MultiRelEdgeDataset(FinancialGraphDataset):
    """Dataset con edge_attr de 3 dimensiones: [|corr|, signo, mismo_sector].

    Attributes:
        same_sector: Matriz `(num_assets, num_assets)` con 1.0 si el par de
            activos comparte sector y 0.0 en caso contrario.
    """

    def __init__(self, data_dir="data", window=60, tau=0.5):
        """Inicializa el dataset base y construye la matriz de mismo-sector.

        Args:
            data_dir: Directorio con los datos preprocesados.
            window: Ventana usada para normalizar precios (ver clase padre).
            tau: Umbral de correlación absoluta para las aristas.

        Notes:
            Si el universo tiene hasta 20 activos, se conserva EXACTAMENTE
            el mapa posicional `_SECTOR_IDS` con que se obtuvieron los
            resultados publicados del cap. 5 (aunque las columnas de
            `data/` estén en orden alfabético, se mantiene la asignación
            posicional original byte a byte). Para universos extendidos
            (p. ej. data_xl, 58 activos) se derivan sectores GICS a partir
            del ticker, en el orden de columnas del dataset.
        """
        super().__init__(data_dir=data_dir, window=window, tau=tau)
        n = self.num_assets
        if n <= len(_SECTOR_IDS):
            sec = _SECTOR_IDS[:n]
        else:
            cats = sorted({_GICS.get(t, t) for t in self.assets})
            cat_id = {c: i for i, c in enumerate(cats)}
            sec = np.array([cat_id[_GICS.get(t, t)] for t in self.assets])
        self.same_sector = (sec[:, None] == sec[None, :]).astype(np.float32)

    def get(self, idx):
        """Genera el grafo G(t) en el instante `idx` con arista multi-relacional.

        Args:
            idx: Índice temporal dentro de `self.valid_dates`.

        Returns:
            Objeto `Data` con `edge_attr` de forma `(E, 3)`, donde cada fila
            es `[|corr_ij|, sign(corr_ij), mismo_sector_ij]`. El resto de
            campos (`x`, `edge_index`, `y`, `cov`) siguen la misma
            construcción que la clase padre.
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
            vals = corr_matrix[indices]
            mag = np.abs(vals).astype(np.float32)
            sign = np.sign(vals).astype(np.float32)
            sect = self.same_sector[indices].astype(np.float32)
            edge_attr = torch.tensor(np.stack([mag, sign, sect], axis=1),
                                     dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)

        vol_t = self.volatility.loc[self.valid_dates[idx]].values
        cov_matrix = corr_matrix * np.outer(vol_t, vol_t)
        cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cov=cov_tensor)


if __name__ == "__main__":
    ds = MultiRelEdgeDataset(tau=0.5)
    s = ds.get(300)
    print("edge_index:", tuple(s.edge_index.shape))
    print("edge_attr :", tuple(s.edge_attr.shape), "(esperado (E, 3))")
    print("muestra edge_attr[:3]:\n", s.edge_attr[:3])
