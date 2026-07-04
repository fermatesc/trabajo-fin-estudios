"""Variante de FinancialGraphDataset que conserva el signo de la correlación.

Se usa para el experimento de comprobar si distinguir correlaciones
negativas (clave de la diversificación) de las positivas mejora el modelo,
en lugar de usar el valor absoluto de la correlación como peso de arista.
"""
import os
import sys

import torch
import numpy as np
from torch_geometric.data import Data

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_utils.dataset_loader import FinancialGraphDataset
else:
    from data_utils.dataset_loader import FinancialGraphDataset


class SignedFinancialGraphDataset(FinancialGraphDataset):
    """Variante de `FinancialGraphDataset` que conserva el signo de la correlación.

    El atributo de arista se define como `edge_attr = corr[indices]`, sin
    aplicar `np.abs`. El filtrado de aristas sigue siendo por `|corr| > tau`
    (mismo grafo que la clase padre), pero el peso de cada arista mantiene
    su signo.

    Hereda la carga de datos y la construcción de features de la clase
    padre (`__init__` / `_build_node_features`) y sobreescribe solamente
    `get(idx)`.
    """

    def get(self, idx):
        """Genera el grafo G(t) en el instante `idx` conservando el signo de la correlación.

        Args:
            idx: Índice temporal dentro de `self.valid_dates`.

        Returns:
            Objeto `Data` idéntico al de la clase padre salvo que
            `edge_attr` conserva el signo de la correlación en vez de su
            valor absoluto.
        """
        x = self.x[idx]

        if idx < self.len() - 1:
            next_date = self.valid_dates[idx + 1]
            y = torch.tensor(self.returns.loc[next_date].values, dtype=torch.float32)
        else:
            y = torch.zeros(self.num_assets, dtype=torch.float32)

        corr_matrix = self.dynamic_corr[idx]

        mask = (np.abs(corr_matrix) > self.tau) & (~np.eye(self.num_assets, dtype=bool))
        indices = np.where(mask)

        if len(indices[0]) > 0:
            edge_index = torch.tensor(np.stack(indices), dtype=torch.long)
            edge_attr = torch.tensor(corr_matrix[indices], dtype=torch.float32).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)

        vol_t = self.volatility.loc[self.valid_dates[idx]].values
        vol_outer = np.outer(vol_t, vol_t)
        cov_matrix = corr_matrix * vol_outer
        cov_tensor = torch.tensor(cov_matrix, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cov=cov_tensor)


if __name__ == "__main__":
    IDX = 50

    try:
        base_ds = FinancialGraphDataset(tau=0.5)
        signed_ds = SignedFinancialGraphDataset(tau=0.5)
    except FileNotFoundError as e:
        print(f"[ERROR] No se pudieron cargar los datos del dataset: {e}")
        print("Asegurate de que los CSV/NPY procesados existen en 'data/'.")
        raise SystemExit(1)

    if len(base_ds) <= IDX:
        print(f"[ERROR] El dataset solo tiene {len(base_ds)} ventanas; idx={IDX} fuera de rango.")
        raise SystemExit(1)

    base = base_ds.get(IDX)
    signed = signed_ds.get(IDX)

    print(f"Comprobaciones sobre get({IDX}) (tau=0.5):")
    print(f"- Aristas base   : {base.num_edges}")
    print(f"- Aristas firmado: {signed.num_edges}")

    same_edge_index = torch.equal(base.edge_index, signed.edge_index)
    print(f"- edge_index IDENTICO entre base y firmado: {same_edge_index}")
    assert same_edge_index, "edge_index difiere: los grafos NO son iguales"

    if signed.num_edges > 0:
        signed_vals = signed.edge_attr.squeeze(1)
        num_neg = int((signed_vals < 0).sum().item())
        min_val = float(signed_vals.min().item())
        if num_neg > 0:
            print(f"- Aristas con correlacion NEGATIVA: {num_neg} (min = {min_val:.4f}) -> el signo se conserva")
        else:
            print(f"- No hay aristas con correlacion negativa por encima del umbral (min = {min_val:.4f}).")
    else:
        print("- No hay aristas en este grafo (num_edges == 0).")

    if base.num_edges > 0:
        abs_match = torch.allclose(signed.edge_attr.abs(), base.edge_attr)
        print(f"- abs(edge_attr firmado) == edge_attr base: {abs_match}")
        assert abs_match, "abs(edge_attr firmado) NO coincide con el edge_attr del base"
    else:
        print("- Sin aristas: comprobacion de magnitudes trivial.")

    print("\nTodas las comprobaciones pasaron correctamente.")
