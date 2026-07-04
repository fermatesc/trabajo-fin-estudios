"""
QGNN_AmplitudeV3: capa de decisión cuántica INTER-ACTIVO (Opción B del TFM).

Diferencia clave con QGNN_Portfolio (qgnn_model.py) y QGNN_V2 (qgnn_v2.py):

    En esas dos arquitecturas el circuito cuántico se evalúa *por nodo*: cada
    activo entra en su propio registro de 4 qubits y el entrelazamiento ocurre
    SOLO entre las 4 features latentes de UN mismo activo (entrelazamiento
    intra-nodo). La cartera, como objeto conjunto, nunca vive en el espacio de
    Hilbert; la mezcla transversal entre activos la hacen el GAT clásico y la
    softmax. Por eso esas variantes son una "no-linealidad por activo" y NO
    contrastan la hipótesis cuántica que de verdad importa.

    QGNN_AmplitudeV3 corrige eso. El front-end clásico (GRU temporal + dos capas
    GAT) es IDÉNTICO al de QGNN_V2 para que la comparación sea limpia, pero la
    cabeza cuántica cambia por completo:

      1. El GAT produce un vector latente por activo; una capa lineal lo reduce a
         UN escalar de "relevancia" por activo  ->  vector s in R^N (N=20).
      2. Ese vector de N scores se codifica como AMPLITUDES de un estado cuántico
         de n_qubits = ceil(log2 N) = 5 qubits (2^5 = 32 >= 20). La cartera entera
         es ahora un único estado conjunto |psi> = sum_i s_i |i>.
      3. StronglyEntanglingLayers entrelaza los 5 qubits: los estados base que
         representan activos DISTINTOS se entrelazan entre sí  ->  ENTRELAZAMIENTO
         INTER-ACTIVO real, en el espacio de Hilbert.
      4. La medición devuelve la distribución de probabilidad de la base
         computacional (regla de Born, qml.probs). Las primeras N probabilidades,
         renormalizadas, SON los pesos de la cartera: long-only y suma 1 por
         construcción (no hace falta softmax).

    Coste: solo 5 qubits => statevector de 32 amplitudes => perfectamente
    simulable y diferenciable (diff_method="backprop"). Es la razón por la que la
    "Opción B" es viable en este universo de 20 activos, a diferencia de la
    codificación de amplitud sobre un índice completo (cientos de activos), que sí
    escala mal.

Pesos cuánticos entrenables: q_layers * n_qubits * 3 = 2 * 5 * 3 = 30 parámetros.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pennylane as qml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class QGNN_AmplitudeV3(nn.Module):
    """Modelo híbrido cuántico-clásico con cabeza cuántica inter-activo basada
    en codificación de amplitud (Opción B del TFM).

    Args:
        n_assets: Número de activos de la cartera (define n_qubits = ceil(log2 N)).
        in_channels: Dimensión de features por nodo (defecto 6, IN_CHANNELS).
        hidden_channels: Dimensión oculta del GAT por cabeza (defecto 8).
        latent_dim: Dimensión latente a la salida del GAT (defecto 4).
        q_layers: Número de StronglyEntanglingLayers del entrelazador (defecto 2).
        gru_hidden: Tamaño del estado oculto de la GRU (defecto 16).
        edge_dim: Dimensión de los atributos de arista (1 base, 3 multirel).

    Notes:
        El VQC se evalúa una vez por grafo (un día), no por nodo. El estado
        conjunto codifica los N activos a la vez, a diferencia de
        QGNN_Portfolio y QGNN_V2 donde el circuito es por nodo.
    """

    def __init__(
        self,
        n_assets: int,
        in_channels: int = 6,
        hidden_channels: int = 8,
        latent_dim: int = 4,
        q_layers: int = 2,
        gru_hidden: int = 16,
        edge_dim: int = 1,
    ):
        super().__init__()

        if n_assets < 2:
            raise ValueError(f"n_assets debe ser >= 2 (recibido {n_assets})")

        self.n_assets = n_assets
        self.latent_dim = latent_dim
        self.q_layers = q_layers
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

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(
                inputs, wires=range(self.n_qubits), pad_with=0.0, normalize=True
            )
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit
        self.q_weights = nn.Parameter(0.05 * torch.randn(q_layers, self.n_qubits, 3))

    def forward(self, x_seq, edge_index, edge_attr=None):
        """Ejecuta el pipeline completo: GRU temporal, GAT, scoring por activo,
        codificación de amplitud y medición (regla de Born).

        Args:
            x_seq: Tensor (lookback, N, in_channels) con la ventana histórica
                de features de cada nodo.
            edge_index: Tensor (2, E) con los índices de las aristas del grafo.
            edge_attr: Tensor (E, edge_dim) con atributos de arista, o None.

        Returns:
            Tensor (N,) con los pesos de cartera, long-only y que suman 1
            (por construcción, vía regla de Born).

        Raises:
            ValueError: Si el número de activos en x_seq no coincide con
                n_assets, ya que la codificación de amplitud fija el número
                de qubits en la construcción del modelo.

        Notes:
            El vector de amplitudes se rellena con ceros hasta 2^n_qubits y se
            le suma un epsilon (1e-6) uniforme para evitar que
            AmplitudeEmbedding divida por cero cuando todos los scores son
            ~0 (p. ej. al inicio del entrenamiento); el sesgo introducido es
            minúsculo. Los pesos del entrelazador (q_weights) se inicializan
            pequeños para que el circuito se comporte ~identidad al arrancar.
        """
        lookback, N, C = x_seq.shape
        if N != self.n_assets:
            raise ValueError(
                f"x_seq trae N={N} activos pero el modelo se construyó con "
                f"n_assets={self.n_assets}. La codificación de amplitud fija el "
                f"nº de qubits, así que N debe ser constante."
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

        amp = F.pad(scores, (0, self.dim - N))
        amp = amp + 1e-6

        probs = self.circuit(amp, self.q_weights)
        probs = probs.to(scores.dtype)

        w = probs[:N]
        w = w / (w.sum() + 1e-12)
        return w


if __name__ == "__main__":
    torch.manual_seed(42)

    LOOKBACK = 20
    N = 20          # nº de activos (coincide con el dataset)
    IN_CH = 6

    print("=" * 65)
    print("QGNN_AmplitudeV3 smoke tests")
    print("=" * 65)

    model = QGNN_AmplitudeV3(
        n_assets=N,
        in_channels=IN_CH,
        hidden_channels=8,
        latent_dim=4,
        q_layers=2,
        gru_hidden=16,
    )
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_quantum = model.q_weights.numel()
    print(f"\n[a] Modelo OK | n_qubits={model.n_qubits} dim={model.dim}")
    print(f"    Parámetros totales={n_total}  (cuánticos={n_quantum})")
    assert model.n_qubits == 5 and model.dim == 32, "Esperado 5 qubits / 32 amplitudes para N=20"

    x_seq = torch.randn(LOOKBACK, N, IN_CH)
    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.rand(edge_index.size(1), 1)
    print(f"[b] x_seq={tuple(x_seq.shape)} edge_index={tuple(edge_index.shape)}")

    model.eval()
    with torch.no_grad():
        w = model(x_seq, edge_index, edge_attr)
    print(f"\n[c] weights shape : {tuple(w.shape)}  (esperado ({N},))")
    print(f"    weights sum   : {w.sum().item():.6f}  (esperado ~1.0)")
    print(f"    weights min   : {w.min().item():.6f}  (esperado >= 0)")
    assert w.shape == (N,), f"Forma incorrecta: {w.shape}"
    assert abs(w.sum().item() - 1.0) < 1e-5, f"Suma != 1: {w.sum().item()}"
    assert w.min().item() >= 0.0, f"Peso negativo: {w.min().item()}"

    model.train()
    w = model(x_seq, edge_index, edge_attr)
    rand_ret = torch.rand(N)
    loss = -torch.dot(w, rand_ret)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"\n[d] Backward loss = {loss.item():.4f}")
    if no_grad:
        print(f"    WARNING — sin gradiente: {no_grad}")
    else:
        print("    Todos los parámetros tienen gradiente.")
    assert not no_grad, f"Parámetros sin gradiente: {no_grad}"
    qgrad = model.q_weights.grad.norm().item()
    print(f"    ||grad q_weights|| = {qgrad:.6e}")
    assert qgrad > 0.0, "El entrelazador cuántico no recibe gradiente"

    print("\n[OK] QGNN_AmplitudeV3 — todas las aserciones pasaron.")

    print("\n" + "=" * 65)
    print("End-to-end con datos reales")
    print("=" * 65)
    from data_utils.sequence_dataset import SequenceGraphDataset

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds = SequenceGraphDataset(data_dir=os.path.join(ROOT, "data"), tau=0.5, lookback=20)
    print(f"num_assets={ds.num_assets}")
    model_real = QGNN_AmplitudeV3(n_assets=ds.num_assets)
    sample = ds.get(50)
    model_real.eval()
    with torch.no_grad():
        w_real = model_real(sample.x_seq, sample.edge_index, sample.edge_attr)
    print(f"  weights sum={w_real.sum().item():.6f} min={w_real.min().item():.6f}")
    assert w_real.shape == (ds.num_assets,)
    assert abs(w_real.sum().item() - 1.0) < 1e-5
    assert w_real.min().item() >= 0.0

    print("\n[OK] End-to-end con datos reales pasado.")
    print("\n" + "=" * 65)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 65)
