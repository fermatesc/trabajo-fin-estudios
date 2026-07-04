"""Arquitectura híbrida QGNN original (GAT clásica + VQC por nodo) para
optimización de carteras."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pennylane as qml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SOFTMAX_TEMPERATURE


class QGNN_Portfolio(nn.Module):
    """Arquitectura híbrida QGNN para optimización de carteras.

    Fase 1: MPNN clásica (Graph Attention Networks) para procesar la
    topología del mercado. Fase 2: VQC (Variational Quantum Circuit) para
    decisión de pesos altamente expresiva.

    Args:
        in_channels: Dimensión de las features de nodo (defecto 6).
        hidden_channels: Dimensión oculta de la GAT por cabeza (defecto 8).
        n_qubits: Número de qubits del circuito cuántico (defecto 4).
        q_layers: Número de StronglyEntanglingLayers (defecto 2).
        temperature: Factor de temperatura de la softmax final (defecto
            config.SOFTMAX_TEMPERATURE).
    """

    def __init__(self, in_channels=6, hidden_channels=8, n_qubits=4, q_layers=2, temperature=SOFTMAX_TEMPERATURE):
        super(QGNN_Portfolio, self).__init__()

        self.n_qubits = n_qubits
        self.temperature = temperature

        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)
        self.gat2 = GATConv(hidden_channels * 2, self.n_qubits, heads=1, concat=False, edge_dim=1)

        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits), rotation='Y')

            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (q_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.fc_out = nn.Linear(self.n_qubits, 1)

    def forward(self, x, edge_index, edge_attr=None):
        """Ejecuta el pipeline completo: GAT clásica, circuito cuántico por
        nodo y softmax de salida.

        Args:
            x: Tensor (Num_Activos, in_channels) con las features de los nodos.
            edge_index: Tensor (2, Num_Aristas) con la conectividad del grafo.
            edge_attr: Tensor (Num_Aristas, 1) con los atributos de las aristas.

        Returns:
            Tensor (Num_Activos,) con los pesos de cartera, suman 1.

        Notes:
            El ansatz StronglyEntanglingLayers es muy expresivo; a
            profundidad L alta favorece la aparición de barren plateaus
            (McClean et al. 2018), por eso la profundidad se mantiene baja
            (ver config.Q_LAYERS). Se aplica tanh antes de AngleEmbedding
            para acotar los vectores latentes a [-1, 1], rango estable para
            los ángulos de rotación. El factor de temperatura en la softmax
            final fuerza a la red a tomar decisiones más agresivas y evita
            que colapse en la estrategia Equal-Weight (1/N).
        """
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = torch.tanh(x)

        q_out = self.qlayer(x)

        out = self.fc_out(q_out)

        weights = F.softmax(out * self.temperature, dim=0).squeeze(-1)

        return weights


if __name__ == "__main__":
    model = QGNN_Portfolio(in_channels=6, n_qubits=4)
    print("Modelo Híbrido QGNN inicializado con éxito:")
    print(model)
