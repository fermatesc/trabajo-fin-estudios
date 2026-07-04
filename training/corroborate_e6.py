"""Corroboración multi-semilla de E6 (n_qubits=6) frente a la base (n_qubits=4).

`experiment_runner.py` ejecutaba cada configuración una sola vez con
seed=42. Aquí se repiten E6 y la base sobre 5 semillas para determinar si
el Sharpe=2.21 obtenido por E6 es un hallazgo real de la configuración o
ruido de inicialización. Escribe los resultados en corroboration_e6.json.
"""
import json
import sys
import os
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_loader import FinancialGraphDataset
from models.qgnn_model import QGNN_Portfolio
from models.loss_functions import DifferentiableSharpeLoss
from evaluation.metrics import calculate_portfolio_metrics
from config import set_global_seed, LEARNING_RATE, IN_CHANNELS

EXP_EPOCHS = 5
SEEDS = [42, 43, 44, 45, 46]

CONFIGS = {
    "BASE_Q4_L2": {"n_qubits": 4, "q_layers": 2, "hidden": 8, "temp": 5.0},
    "E6_Q6_L2":   {"n_qubits": 6, "q_layers": 2, "hidden": 8, "temp": 5.0},
}


def run_once(cfg, seed, dataset, train_size, test_indices):
    """Entrena y backtestea una configuración de CONFIGS con una semilla dada.

    Args:
        cfg: Diccionario de hiperparámetros (n_qubits, q_layers, hidden, temp).
        seed: Semilla global para esta ejecución.
        dataset: FinancialGraphDataset del que se extraen los grafos.
        train_size: Número de días de entrenamiento.
        test_indices: Índices de test sobre `dataset`.

    Returns:
        dict: Métricas de cartera devueltas por calculate_portfolio_metrics.
    """
    set_global_seed(seed)
    model = QGNN_Portfolio(in_channels=IN_CHANNELS,
                           hidden_channels=cfg["hidden"],
                           n_qubits=cfg["n_qubits"],
                           q_layers=cfg["q_layers"],
                           temperature=cfg["temp"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DifferentiableSharpeLoss()

    for epoch in range(EXP_EPOCHS):
        model.train()
        for i in range(train_size):
            data = dataset[i]
            optimizer.zero_grad()
            weights = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(weights, data.y, data.cov)
            loss.backward()
            optimizer.step()

    model.eval()
    rets = []
    with torch.no_grad():
        for i in test_indices:
            data = dataset[i]
            weights = model(data.x, data.edge_index, data.edge_attr)
            rets.append(np.dot(weights.numpy(), data.y.numpy()))
    return calculate_portfolio_metrics(np.array(rets))


def main():
    """Corre BASE_Q4_L2 y E6_Q6_L2 sobre SEEDS y guarda el resumen en JSON."""
    dataset = FinancialGraphDataset(tau=0.5)
    total = len(dataset)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)
    test_indices = range(train_size + val_size, total)

    out = {}
    for name, cfg in CONFIGS.items():
        sharpes, rets_an = [], []
        for seed in SEEDS:
            m = run_once(cfg, seed, dataset, train_size, test_indices)
            sharpes.append(m["sharpe_ratio"])
            rets_an.append(m["retorno_anualizado"])
            print(f"{name} seed={seed}: Sharpe={m['sharpe_ratio']:.3f}  Ret={m['retorno_anualizado']:.2%}", flush=True)
        out[name] = {
            "sharpe_per_seed": sharpes,
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_min": float(np.min(sharpes)),
            "sharpe_max": float(np.max(sharpes)),
            "ret_anual_mean": float(np.mean(rets_an)),
        }
        print(f"==> {name}: mean={out[name]['sharpe_mean']:.3f} +/- {out[name]['sharpe_std']:.3f} "
              f"[{out[name]['sharpe_min']:.3f}, {out[name]['sharpe_max']:.3f}]\n", flush=True)

    with open("corroboration_e6.json", "w") as f:
        json.dump(out, f, indent=4)
    print("Guardado en corroboration_e6.json")


if __name__ == "__main__":
    main()
