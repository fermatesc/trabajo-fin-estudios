"""Envoltorio de FinancialGraphDataset para secuencias temporales con ventana deslizante.

Expone secuencias de longitud `lookback` sobre los grafos diarios de
`FinancialGraphDataset` sin duplicar ninguna lógica de carga de datos.
"""
import os
import sys
import torch
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_loader import FinancialGraphDataset


class SequenceGraphDataset:
    """Compone `FinancialGraphDataset` para ofrecer secuencias temporales de grafos.

    Para cada índice válido `idx` (dentro de `valid_range`), `get(idx)` devuelve:
      - x_seq: features de nodo, forma ``(lookback, N, 6)``, de los días
        ``[idx-lookback+1 .. idx]``.
      - edge_index, edge_attr, y, cov: topología y targets del día `idx`
        (obtenidos de `base.get`).

    Attributes:
        base: Instancia de `FinancialGraphDataset` subyacente.
        lookback: Longitud de la ventana temporal de la secuencia.
        num_assets: Número de activos.
    """

    def __init__(self, data_dir="data", tau=0.5, lookback=20):
        """Inicializa el dataset base y guarda los parámetros de la secuencia.

        Args:
            data_dir: Directorio con los datos preprocesados.
            tau: Umbral de correlación absoluta para las aristas del grafo base.
            lookback: Longitud de la ventana temporal de cada secuencia.
        """
        self.base = FinancialGraphDataset(data_dir=data_dir, tau=tau)
        self.lookback = lookback
        self.num_assets = self.base.num_assets

    def __len__(self):
        """Número total de instantes temporales del dataset base."""
        return len(self.base)

    def valid_range(self):
        """Devuelve el rango de índices que disponen de ventana `lookback` completa.

        Returns:
            Objeto `range` cuyo primer valor es `lookback - 1` (necesita los
            días 0 .. lookback-1) y cuyo último valor es `len(self.base) - 1`.
        """
        return range(self.lookback - 1, len(self.base))

    def get(self, idx):
        """Construye una muestra de secuencia centrada en el instante `idx`.

        Args:
            idx: Índice temporal; debe cumplir `idx >= lookback - 1`.

        Returns:
            Objeto `torch_geometric.data.Data` con el atributo adicional `x_seq`:
                x_seq: Tensor ``(lookback, N, 6)``.
                edge_index: Tensor ``(2, E)``.
                edge_attr: Tensor ``(E, 1)``.
                y: Tensor ``(N,)``.
                cov: Tensor ``(N, N)``.

        Raises:
            IndexError: Si `idx` es menor que `lookback - 1`.

        Notes:
            `x_seq` se obtiene troceando el tensor `self.base.x` ya cargado en
            memoria (forma ``(T, N, 6)``), sin I/O adicional.
        """
        if idx < self.lookback - 1:
            raise IndexError(
                f"idx={idx} is too small for lookback={self.lookback}. "
                f"First valid index is {self.lookback - 1}."
            )

        start = idx - self.lookback + 1
        x_seq = self.base.x[start : idx + 1]

        day_data = self.base.get(idx)

        return Data(
            x_seq=x_seq,
            edge_index=day_data.edge_index,
            edge_attr=day_data.edge_attr,
            y=day_data.y,
            cov=day_data.cov,
        )

    def __getitem__(self, idx):
        """Alias de `get(idx)` para soportar indexación estándar."""
        return self.get(idx)


if __name__ == "__main__":
    import sys

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT, "data")

    print("=" * 60)
    print("SequenceGraphDataset smoke test")
    print("=" * 60)

    ds = SequenceGraphDataset(data_dir=DATA_DIR, tau=0.5, lookback=20)
    print(f"Base dataset length : {len(ds)}")
    print(f"Valid range         : {ds.valid_range().start} .. {ds.valid_range().stop - 1}")
    print(f"Num assets          : {ds.num_assets}")

    sample = ds.get(50)
    print(f"\nSample at idx=50:")
    print(f"  x_seq shape   : {tuple(sample.x_seq.shape)}  (expected (20, {ds.num_assets}, 6))")
    print(f"  edge_index    : {tuple(sample.edge_index.shape)}")
    print(f"  edge_attr     : {tuple(sample.edge_attr.shape)}")
    print(f"  y             : {tuple(sample.y.shape)}")
    print(f"  cov           : {tuple(sample.cov.shape)}")

    assert sample.x_seq.shape == (20, ds.num_assets, 6), "x_seq shape mismatch"
    assert sample.y.shape == (ds.num_assets,), "y shape mismatch"
    assert sample.cov.shape == (ds.num_assets, ds.num_assets), "cov shape mismatch"
    print("\n[OK] All shape assertions passed.")
