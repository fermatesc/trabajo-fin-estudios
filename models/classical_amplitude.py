"""
ClassicalAmplitudeV3: gemelo clásico CONJUNTO de QGNN_AmplitudeV3 (qgnn_amplitude.py).

Por qué este gemelo y no ClassicalGNN_V2:

    ClassicalGNN_V2 mezcla los activos solo en el GAT y decide los pesos con una
    cabeza POR NODO (MLP latent->latent aplicado a cada activo por separado). Es el
    contrafactual correcto del QGNN_V2 *por nodo*, pero NO del modelo de amplitud:
    QGNN_AmplitudeV3 mezcla los N activos de forma CONJUNTA en el espacio de Hilbert
    (entrelazamiento inter-activo) y decide por regla de Born.

    Para aislar "¿aporta algo la mezcla cuántica conjunta?" el gemelo debe mezclar
    también de forma conjunta, pero por medios clásicos. ClassicalAmplitudeV3 hace
    exactamente eso: comparte front-end (GRU + GAT) con el modelo cuántico, reduce a
    un vector de N scores y lo pasa por una capa de mezcla DENSA sobre el vector
    completo (toca todos los activos a la vez), seguida de softmax. La única
    diferencia con el cuántico es el mecanismo de mezcla:

        cuántico  : amplitudes -> unitario entrelazante U(theta) -> Born |.|^2
        clásico   : scores     -> mezcla densa W -> softmax

    Conteo de parámetros de la cabeza de mezcla (dim = 2^ceil(log2 N); N=20 -> dim=32):
        - mix_hidden=None : Linear(dim, dim)               = dim*dim + dim = 1056
        - mix_hidden=k    : Linear(dim,k)+Linear(k,dim)    = 2*dim*k + k + dim
    La cabeza cuántica equivalente tiene q_layers*n_qubits*3 = 30 parámetros. Es
    decir, el gemelo clásico dispone de MÁS capacidad: si aun así no gana el
    cuántico, la comparación es CONSERVADORA a favor de lo clásico (misma postura
    que el resto de la tesis con los gemelos dimension-matched).
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ClassicalAmplitudeV3(nn.Module):
    """Gemelo clásico conjunto de QGNN_AmplitudeV3: mismo front-end (GRU + GAT)
    pero con una mezcla densa clásica sobre el vector completo de scores en
    lugar del entrelazamiento cuántico.

    Args:
        n_assets: Número de activos (fija dim = 2^ceil(log2 N), igual que el
            modelo cuántico).
        in_channels: Dimensión de features por nodo (defecto 6).
        hidden_channels: Dimensión oculta del GAT por cabeza (defecto 8).
        latent_dim: Dimensión latente a la salida del GAT (defecto 4).
        gru_hidden: Tamaño del estado oculto de la GRU (defecto 16).
        temperature: Nitidez de la softmax (defecto 1.5).
        mix_hidden: Si es None, mezcla densa plena Linear(dim, dim); si es
            int, cuello de botella Linear(dim, k) + Tanh + Linear(k, dim).
        edge_dim: Dimensión de los atributos de arista (1 base, 3 multirel).
    """

    def __init__(
        self,
        n_assets: int,
        in_channels: int = 6,
        hidden_channels: int = 8,
        latent_dim: int = 4,
        gru_hidden: int = 16,
        temperature: float = 1.5,
        mix_hidden=None,
        edge_dim: int = 1,
    ):
        super().__init__()

        if n_assets < 2:
            raise ValueError(f"n_assets debe ser >= 2 (recibido {n_assets})")

        self.n_assets = n_assets
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.edge_dim = edge_dim
        self.n_qubits = max(1, math.ceil(math.log2(n_assets)))
        self.dim = 2 ** self.n_qubits

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

        self.fc_score = nn.Linear(latent_dim, 1)

        if mix_hidden is None:
            self.mixer = nn.Linear(self.dim, self.dim)
        else:
            self.mixer = nn.Sequential(
                nn.Linear(self.dim, mix_hidden),
                nn.Tanh(),
                nn.Linear(mix_hidden, self.dim),
            )

    def forward(self, x_seq, edge_index, edge_attr=None):
        """Ejecuta el pipeline completo: GRU temporal, GAT, scoring por activo,
        mezcla densa conjunta y softmax.

        Args:
            x_seq: Tensor (lookback, N, in_channels) con la ventana histórica
                de features de cada nodo.
            edge_index: Tensor (2, E) con los índices de las aristas del grafo.
            edge_attr: Tensor (E, edge_dim) con atributos de arista, o None.

        Returns:
            Tensor (N,) con los pesos de cartera, long-only y que suman 1 (softmax).

        Raises:
            ValueError: Si el número de activos en x_seq no coincide con
                n_assets, ya que la dimensión de la mezcla queda fija en la
                construcción del modelo.
        """
        lookback, N, C = x_seq.shape
        if N != self.n_assets:
            raise ValueError(
                f"x_seq trae N={N} pero el modelo se construyó con n_assets="
                f"{self.n_assets}; el gemelo de amplitud fija dim, N debe ser fijo."
            )

        x_node = x_seq.permute(1, 0, 2)
        _, h_n = self.gru(x_node)
        h = h_n.squeeze(0)

        h = self.gat1(h, edge_index, edge_attr=edge_attr)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = torch.tanh(h)

        scores = self.fc_score(h).squeeze(-1)
        v = F.pad(scores, (0, self.dim - N))

        mixed = self.mixer(v)
        logits = mixed[:N]
        weights = F.softmax(self.temperature * logits, dim=0)
        return weights


if __name__ == "__main__":
    torch.manual_seed(42)

    LOOKBACK, N, IN_CH = 20, 20, 6

    print("=" * 65)
    print("ClassicalAmplitudeV3 smoke tests")
    print("=" * 65)

    model = ClassicalAmplitudeV3(n_assets=N, in_channels=IN_CH, latent_dim=4)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_mix = sum(p.numel() for p in model.mixer.parameters())
    print(f"\n[a] Modelo OK | dim={model.dim} | params totales={n_total} (mezcla={n_mix})")

    x_seq = torch.randn(LOOKBACK, N, IN_CH)
    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.rand(edge_index.size(1), 1)

    model.eval()
    with torch.no_grad():
        w = model(x_seq, edge_index, edge_attr)
    print(f"\n[b] weights sum={w.sum().item():.6f} min={w.min().item():.6f}")
    assert w.shape == (N,)
    assert abs(w.sum().item() - 1.0) < 1e-5
    assert w.min().item() >= 0.0

    model.train()
    w = model(x_seq, edge_index, edge_attr)
    loss = -torch.dot(w, torch.rand(N))
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"[c] Backward OK, sin gradiente: {no_grad if no_grad else 'ninguno'}")
    assert not no_grad

    model_bn = ClassicalAmplitudeV3(n_assets=N, mix_hidden=5)
    n_mix_bn = sum(p.numel() for p in model_bn.mixer.parameters())
    print(f"[d] Variante mix_hidden=5 -> params mezcla={n_mix_bn}")

    print("\n[OK] ClassicalAmplitudeV3 — todas las aserciones pasaron.")
    print("=" * 65)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 65)
