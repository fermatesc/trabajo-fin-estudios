"""
Script de experimentación para comparar escenarios de hiperparámetros.

Ejecuta entrenamiento + backtesting sobre una lista de configuraciones
(EXPERIMENTS) y acumula los resultados en results_experiments.json.
"""
import json
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_loader import FinancialGraphDataset
from models.qgnn_model import QGNN_Portfolio
from models.classical_gnn_model import ClassicalGNN_Portfolio
from models.markowitz_baseline import MarkowitzOptimizer
from models.loss_functions import DifferentiableSharpeLoss
from evaluation.metrics import calculate_portfolio_metrics
from config import set_global_seed, SEED, LEARNING_RATE, IN_CHANNELS

EXP_EPOCHS = 5

EXPERIMENTS = [
    {"id": "E1_L2", "model": "qgnn", "n_qubits": 4, "q_layers": 2, "hidden": 8, "temp": 5.0, "learnable_temp": False},
    {"id": "E2_L4", "model": "qgnn", "n_qubits": 4, "q_layers": 4, "hidden": 8, "temp": 5.0, "learnable_temp": False},
    {"id": "E3_L6", "model": "qgnn", "n_qubits": 4, "q_layers": 6, "hidden": 8, "temp": 5.0, "learnable_temp": False},
    {"id": "E4_L8", "model": "qgnn", "n_qubits": 4, "q_layers": 8, "hidden": 8, "temp": 5.0, "learnable_temp": False},
    {"id": "E5_L10", "model": "qgnn", "n_qubits": 4, "q_layers": 10, "hidden": 8, "temp": 5.0, "learnable_temp": False},
    {"id": "E6_Q6_L2", "model": "qgnn", "n_qubits": 6, "q_layers": 2, "hidden": 8, "temp": 5.0, "learnable_temp": False},
    {"id": "E7_LearnTemp", "model": "qgnn", "n_qubits": 4, "q_layers": 2, "hidden": 8, "temp": 5.0, "learnable_temp": True},
    {"id": "B1", "model": "equal_weight"},
    {"id": "B2", "model": "markowitz"},
    {"id": "B3", "model": "classical_gnn", "hidden": 8, "latent_dim": 4, "temp": 5.0},
]


def run_experiment(exp, dataset, train_size, val_size, test_indices):
    """Entrena (si aplica) y backtestea una configuración de EXPERIMENTS.

    Según `exp["model"]` puede ser una estrategia sin entrenamiento
    (Equal-Weight, Markowitz) o un modelo neuronal (QGNN, GNN clásica)
    entrenado con DifferentiableSharpeLoss durante EXP_EPOCHS épocas.

    Args:
        exp: Diccionario de configuración del experimento (una entrada de
            EXPERIMENTS), con al menos la clave "model".
        dataset: FinancialGraphDataset del que se extraen los grafos.
        train_size: Número de días de entrenamiento.
        val_size: Número de días de validación (no usado dentro de esta
            función, se recibe por consistencia con el resto del harness).
        test_indices: Índices de test sobre `dataset`.

    Returns:
        dict: Métricas de cartera devueltas por calculate_portfolio_metrics.
    """
    set_global_seed(SEED)
    model_type = exp["model"]

    if model_type == "equal_weight":
        ew_weights = torch.ones(dataset.num_assets) / dataset.num_assets
        rets = []
        for i in test_indices:
            rets.append(np.dot(ew_weights.numpy(), dataset[i].y.numpy()))
        return calculate_portfolio_metrics(np.array(rets))
        
    elif model_type == "markowitz":
        markowitz = MarkowitzOptimizer()
        rets = []
        for i in test_indices:
            idx_date = dataset.valid_dates[i]
            window_returns = dataset.returns.loc[:idx_date].iloc[-60:].values
            w_marko = markowitz.optimize(window_returns)
            rets.append(np.dot(w_marko, dataset[i].y.numpy()))
        return calculate_portfolio_metrics(np.array(rets))
        
    if model_type == "qgnn":
        model = QGNN_Portfolio(in_channels=IN_CHANNELS, 
                               hidden_channels=exp["hidden"], 
                               n_qubits=exp["n_qubits"], 
                               q_layers=exp["q_layers"],
                               temperature=exp["temp"])
        if exp["learnable_temp"]:
            model.temperature = torch.nn.Parameter(torch.tensor(exp["temp"]))
            
    elif model_type == "classical_gnn":
        model = ClassicalGNN_Portfolio(in_channels=IN_CHANNELS, 
                                       hidden_channels=exp["hidden"], 
                                       latent_dim=exp["latent_dim"], 
                                       temperature=exp["temp"])
                                       
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DifferentiableSharpeLoss()
    
    print(f"Entrenando {exp['id']}...")
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
    """Ejecuta todas las configuraciones de EXPERIMENTS y guarda los resultados en JSON."""
    print("Iniciando Experimentación de Hiperparámetros...")
    dataset = FinancialGraphDataset(tau=0.5)
    total_samples = len(dataset)
    train_size = int(total_samples * 0.70)
    val_size = int(total_samples * 0.15)
    test_start = train_size + val_size
    test_indices = range(test_start, total_samples)
    
    results = {}
    
    for exp in EXPERIMENTS:
        print(f"\n================ Ejecutando {exp['id']} ================")
        try:
            res = run_experiment(exp, dataset, train_size, val_size, test_indices)
            results[exp["id"]] = res
            print(f"Resultado {exp['id']}: Sharpe={res['sharpe_ratio']:.3f}, Retorno={res['retorno_anualizado']:.2%}")
        except Exception as e:
            print(f"Error en {exp['id']}: {str(e)}")

    with open("results_experiments.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResultados guardados en results_experiments.json")


if __name__ == "__main__":
    main()
