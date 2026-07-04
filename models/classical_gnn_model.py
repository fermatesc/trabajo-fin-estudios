"""Modelo GNN clásico (baseline) para comparación con el QGNN híbrido.

Arquitectura idéntica al QGNN pero sustituyendo la capa cuántica por un MLP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ClassicalGNN_Portfolio(nn.Module):
    """Baseline clásico: GAT + MLP (sin circuito cuántico).

    Mantiene la misma dimensionalidad que el QGNN para comparación justa.

    Args:
        in_channels: Dimensión de las features de nodo (defecto 6).
        hidden_channels: Dimensión oculta de la GAT por cabeza (defecto 8).
        latent_dim: Dimensión latente, equivalente clásico de n_qubits en
            el QGNN (defecto 4).
        temperature: Factor de temperatura de la softmax final (defecto 5.0).

    Notes:
        El MLP tiene dimensionalidad latente equivalente al VQC (4 -> 4).
        Conteo de parámetros del bloque: el VQC (q_layers=2, n_qubits=4)
        tiene 2*4*3 = 24 parámetros, mientras que este MLP tiene
        4*8+8 + 8*4+4 = 76 parámetros. La diferencia favorece al baseline
        clásico (comparación conservadora, declarada en la memoria del TFM).
    """

    def __init__(self, in_channels=6, hidden_channels=8, latent_dim=4, temperature=5.0):
        super(ClassicalGNN_Portfolio, self).__init__()

        self.temperature = temperature
        self.latent_dim = latent_dim

        self.gat1 = GATConv(in_channels, hidden_channels, heads=2, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 2)
        self.gat2 = GATConv(hidden_channels * 2, latent_dim, heads=1, concat=False, edge_dim=1)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.Tanh(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )

        self.fc_out = nn.Linear(latent_dim, 1)

    def forward(self, x, edge_index, edge_attr=None):
        """Ejecuta el pipeline completo: GAT clásica, MLP (en vez de VQC) y
        softmax de salida.

        Args:
            x: Tensor (Num_Activos, in_channels) con las features de los nodos.
            edge_index: Tensor (2, Num_Aristas) con la conectividad del grafo.
            edge_attr: Tensor (Num_Aristas, 1) con los atributos de las aristas.

        Returns:
            Tensor (Num_Activos,) con los pesos de cartera, suman 1.
        """
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        x = torch.tanh(x)

        x = self.mlp(x)

        out = self.fc_out(x)
        weights = F.softmax(out * self.temperature, dim=0).squeeze(-1)

        return weights


if __name__ == "__main__":
    model = ClassicalGNN_Portfolio(in_channels=6, latent_dim=4)
    print("Modelo GNN Clásico (Baseline) inicializado:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros entrenables: {total_params}")
