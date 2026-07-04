"""Modelo híbrido cuántico-clásico QGNN_V2 para optimización de carteras.

Introduce tres mejoras respecto a QGNN_Portfolio (qgnn_model.py):
1. Codificador temporal: una GRU por nodo agrega una ventana de lookback
   antes de la GAT.
2. Data re-uploading: las features cuánticas se reinyectan antes de cada
   capa de entrelazamiento (Pérez-Salinas et al. 2020), aumentando la
   expresividad efectiva sin profundizar el circuito.
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
import pennylane as qml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class QGNN_V2(nn.Module):
    """Modelo híbrido cuántico-clásico con codificador temporal, GAT, data
    re-uploading y salida residual sobre Equal-Weight.

    Args:
        in_channels: Dimensión de las features de nodo (defecto 6, coincide
            con IN_CHANNELS en config).
        hidden_channels: Dimensión oculta de la GAT por cabeza (defecto 8).
        n_qubits: Anchura del circuito = dimensión latente cuántica (defecto 4).
        q_layers: Número de StronglyEntanglingLayers (defecto 2).
        gru_hidden: Tamaño del estado oculto de la GRU (defecto 16).
        temperature: Nitidez del softmax; 1.5 ≈ concentración moderada (defecto 1.5).
        reupload: Activa el data re-uploading (defecto True).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 8,
        n_qubits: int = 4,
        q_layers: int = 2,
        gru_hidden: int = 16,
        temperature: float = 1.5,
        reupload: bool = True,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.q_layers = q_layers
        self.temperature = temperature
        self.reupload = reupload
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
            hidden_channels * 2, n_qubits, heads=1, concat=False, edge_dim=edge_dim
        )

        dev = qml.device("default.qubit", wires=n_qubits)

        if reupload:
            @qml.qnode(dev, interface="torch")
            def quantum_circuit(inputs, weights):
                for l in range(q_layers):
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                    qml.StronglyEntanglingLayers(
                        weights[l : l + 1], wires=range(n_qubits)
                    )
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        else:
            @qml.qnode(dev, interface="torch")
            def quantum_circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (q_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.fc_out = nn.Linear(n_qubits, 1)
        nn.init.normal_(self.fc_out.weight, std=0.01)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x_seq, edge_index, edge_attr=None):
        """Ejecuta el pipeline completo: GRU temporal, GAT, circuito cuántico
        con re-uploading y salida residual sobre 1/N.

        Args:
            x_seq: Tensor (lookback, N, in_channels) con la ventana histórica
                de features de cada nodo.
            edge_index: Tensor (2, E) con los índices de las aristas del grafo.
            edge_attr: Tensor (E, 1) con atributos de arista, o None.

        Returns:
            Tensor (N,) con los pesos de cartera, long-only y que suman 1.

        Notes:
            Se aplica ``tanh`` antes del ``AngleEmbedding`` para acotar las
            features a [-1, 1], el rango natural de una rotación de ángulo.
            La salida se centra restando la media (delta=0 implica Equal-Weight)
            y el peso ``fc_out`` se inicializa con std=0.01 para que la cartera
            arranque cerca de EW en lugar de una esquina arbitraria del simplex.
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

        q_out = self.qlayer(h)

        delta = self.fc_out(q_out).squeeze(-1)
        delta = delta - delta.mean()
        weights = F.softmax(self.temperature * delta, dim=0)

        return weights


if __name__ == "__main__":
    import math

    torch.manual_seed(42)

    LOOKBACK = 20
    N = 20          # number of assets (matches dataset)
    IN_CH = 6

    print("=" * 65)
    print("QGNN_V2 smoke tests")
    print("=" * 65)

    model = QGNN_V2(
        in_channels=IN_CH,
        hidden_channels=8,
        n_qubits=4,
        q_layers=2,
        gru_hidden=16,
        temperature=1.5,
        reupload=True,
    )
    print(f"\n[a] Model built OK: {sum(p.numel() for p in model.parameters())} params")

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

    print("\n[OK] QGNN_V2 all assertions passed.")

    print("\n" + "=" * 65)
    print("SequenceGraphDataset smoke test (real data)")
    print("=" * 65)

    import os

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT, "data")

    from data_utils.sequence_dataset import SequenceGraphDataset

    ds = SequenceGraphDataset(data_dir=DATA_DIR, tau=0.5, lookback=20)
    print(f"Base length  : {len(ds)}")
    print(f"Num assets   : {ds.num_assets}")
    print(f"Valid range  : {ds.valid_range().start} .. {ds.valid_range().stop - 1}")

    sample = ds.get(50)
    print(f"\nSample idx=50:")
    print(f"  x_seq      : {tuple(sample.x_seq.shape)}   (expected (20, {ds.num_assets}, 6))")
    print(f"  edge_index : {tuple(sample.edge_index.shape)}")
    print(f"  edge_attr  : {tuple(sample.edge_attr.shape)}")
    print(f"  y          : {tuple(sample.y.shape)}")
    print(f"  cov        : {tuple(sample.cov.shape)}")

    assert sample.x_seq.shape == (20, ds.num_assets, 6)
    assert sample.y.shape == (ds.num_assets,)
    assert sample.cov.shape == (ds.num_assets, ds.num_assets)

    print("\n[OK] SequenceGraphDataset all assertions passed.")

    print("\n" + "=" * 65)
    print("End-to-end forward with real sequence sample")
    print("=" * 65)

    model_real = QGNN_V2(
        in_channels=6,
        hidden_channels=8,
        n_qubits=4,
        q_layers=2,
        gru_hidden=16,
        temperature=1.5,
        reupload=True,
    )
    model_real.eval()
    with torch.no_grad():
        w_real = model_real(
            sample.x_seq,
            sample.edge_index,
            sample.edge_attr,
        )

    print(f"  weights shape : {tuple(w_real.shape)}")
    print(f"  weights sum   : {w_real.sum().item():.6f}")
    print(f"  weights min   : {w_real.min().item():.6f}")
    assert w_real.shape == (ds.num_assets,)
    assert abs(w_real.sum().item() - 1.0) < 1e-5
    assert w_real.min().item() >= 0.0

    print("\n[OK] End-to-end forward with real data passed.")
    print("\n" + "=" * 65)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 65)
