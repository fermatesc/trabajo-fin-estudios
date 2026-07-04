"""Gemelo clásico de QGNN_V2 (qgnn_v2.py) para la comparación cuántico-clásica.

Misma arquitectura que QGNN_V2 pero sustituyendo el circuito cuántico (VQC con
data re-uploading) por un MLP de dimensión equivalente, de modo que ambos
modelos puedan compararse de forma justa. El codificador temporal, la pila GAT
y la salida residual sobre 1/N son idénticos; solo difiere el paso 3
(la transformación latente).

Comparte tres características con QGNN_V2:
1. Codificador temporal: una GRU por nodo agrega una ventana de lookback
   antes de la GAT.
2. Transformación latente: en vez del VQC, un MLP mapea (N, latent_dim) ->
   (N, latent_dim) con el mismo estilo que el MLP de ClassicalGNN_Portfolio.
3. Salida residual sobre 1/N: la FC final produce un delta centrado en 0,
   de modo que en la inicialización la cartera arranca cerca de
   Equal-Weight (1/N) en lugar de colapsar a una esquina arbitraria.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ClassicalGNN_V2(nn.Module):
    """Gemelo clásico de QGNN_V2: mismo front-end (GRU + GAT) pero con un MLP
    en lugar del circuito cuántico.

    Args:
        in_channels: Dimensión de las features de nodo (defecto 6, coincide
            con IN_CHANNELS en config).
        hidden_channels: Dimensión oculta de la GAT por cabeza (defecto 8).
        latent_dim: Dimensión latente, equivalente clásico de n_qubits en
            QGNN_V2 (defecto 4).
        gru_hidden: Tamaño del estado oculto de la GRU (defecto 16).
        temperature: Nitidez del softmax; 1.5 ≈ concentración moderada (defecto 1.5).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 8,
        latent_dim: int = 4,
        gru_hidden: int = 16,
        temperature: float = 1.5,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.temperature = temperature
        self.edge_dim = edge_dim

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.gat1 = GATConv(
            gru_hidden, hidden_channels, heads=2, concat=True, edge_dim=edge_dim
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)

        self.gat2 = GATConv(
            hidden_channels * 2, latent_dim, heads=1, concat=False, edge_dim=edge_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
        )

        self.fc_out = nn.Linear(latent_dim, 1)
        nn.init.normal_(self.fc_out.weight, std=0.01)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x_seq, edge_index, edge_attr=None):
        """Ejecuta el pipeline completo: GRU temporal, GAT, MLP (en lugar del
        circuito cuántico) y salida residual sobre 1/N.

        Args:
            x_seq: Tensor (lookback, N, in_channels) con la ventana histórica
                de features de cada nodo.
            edge_index: Tensor (2, E) con los índices de las aristas del grafo.
            edge_attr: Tensor (E, 1) con atributos de arista, o None.

        Returns:
            Tensor (N,) con los pesos de cartera, long-only y que suman 1.
        """
        lookback, N, C = x_seq.shape

        x_node = x_seq.permute(1, 0, 2)
        _, h_n = self.gru(x_node)
        h = h_n.squeeze(0)

        h = self.gat1(h, edge_index, edge_attr=edge_attr)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = torch.tanh(h)

        c_out = self.mlp(h)

        delta = self.fc_out(c_out).squeeze(-1)
        delta = delta - delta.mean()
        weights = F.softmax(self.temperature * delta, dim=0)

        return weights


if __name__ == "__main__":
    torch.manual_seed(42)

    LOOKBACK = 20
    N = 20
    IN_CH = 6

    print("=" * 65)
    print("ClassicalGNN_V2 smoke tests")
    print("=" * 65)

    model = ClassicalGNN_V2(
        in_channels=IN_CH,
        hidden_channels=8,
        latent_dim=4,
        gru_hidden=16,
        temperature=1.5,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[a] Model built OK: {n_params} trainable params")

    x_seq = torch.randn(LOOKBACK, N, IN_CH)

    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.rand(edge_index.size(1), 1)

    print(f"[b] x_seq={tuple(x_seq.shape)}, edge_index={tuple(edge_index.shape)}, "
          f"edge_attr={tuple(edge_attr.shape)}")

    model.eval()
    with torch.no_grad():
        w = model(x_seq, edge_index, edge_attr)

    print(f"\n[c] weights shape : {tuple(w.shape)}  (expected ({N},))")
    print(f"    weights sum   : {w.sum().item():.6f}  (expected ~1.0)")
    print(f"    weights min   : {w.min().item():.6f}  (expected >= 0)")

    assert w.shape == (N,), f"Shape mismatch: {w.shape}"
    assert abs(w.sum().item() - 1.0) < 1e-5, f"Sum != 1: {w.sum().item()}"
    assert w.min().item() >= 0.0, f"Negative weight: {w.min().item()}"

    ew = 1.0 / N
    max_dev = (w - ew).abs().max().item()
    print(f"\n[d] Equal-weight baseline : {ew:.4f}")
    print(f"    Max deviation from 1/N : {max_dev:.6f}  (should be small)")
    assert max_dev < 0.10, f"Deviation from EW too large at init: {max_dev:.4f}"

    model.train()
    w = model(x_seq, edge_index, edge_attr)
    rand_ret = torch.rand(N)
    loss = -torch.dot(w, rand_ret)
    loss.backward()

    params_with_grad = [
        (name, p.grad is not None)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    no_grad = [name for name, has in params_with_grad if not has]
    print(f"\n[e] Backward loss = {loss.item():.4f}")
    if no_grad:
        print(f"    WARNING — no grad: {no_grad}")
    else:
        print(f"    All {len(params_with_grad)} trainable params have gradients.")
    assert not no_grad, f"Some params lack gradients: {no_grad}"

    print("\n[OK] ClassicalGNN_V2 all assertions passed.")
    print("\n" + "=" * 65)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 65)
