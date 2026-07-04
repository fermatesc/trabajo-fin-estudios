"""Analiza la rotación temporal y la concentración sectorial de los pesos
predichos por el modelo QGNN a lo largo de todo el histórico del dataset.

Genera ``data/weights_history.csv`` con la serie temporal de pesos por activo.
"""
import sys
import os

import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_loader import FinancialGraphDataset
from models.qgnn_model import QGNN_Portfolio
from config import IN_CHANNELS, HIDDEN_CHANNELS, N_QUBITS, Q_LAYERS, SEED, set_global_seed


def analyze_weights():
    """Carga el modelo QGNN de la semilla de referencia, infiere los pesos de
    cartera para todo el histórico y exporta el resultado a CSV.

    Notes:
        Además exporta el CSV, calcula la concentración media en el sector
        Energía (XOM, CVX, COP, SLB) durante 2022 si el rango de fechas del
        dataset cubre ese año, como caso de estudio de rotación sectorial.
    """
    print("Iniciando análisis de rotación de pesos (QGNN)...")

    set_global_seed(SEED)

    dataset = FinancialGraphDataset(tau=0.5)

    model = QGNN_Portfolio(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, n_qubits=N_QUBITS, q_layers=Q_LAYERS)
    try:
        model.load_state_dict(torch.load(f"qgnn_model_seed_{SEED}.pth"))
        print(f"Pesos del modelo (semilla {SEED}) cargados correctamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró qgnn_model_seed_{SEED}.pth. Ejecute el entrenamiento primero.")
        return

    model.eval()

    weights_history = []
    dates = []

    print("Infirendo pesos para todo el histórico...")
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            w = model(data.x, data.edge_index, data.edge_attr).numpy()
            weights_history.append(w)
            dates.append(dataset.valid_dates[i])

    df_weights = pd.DataFrame(weights_history, index=dates, columns=dataset.returns.columns)

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "weights_history.csv")
    df_weights.to_csv(out_path)

    print(f"Historia de pesos exportada a '{out_path}'.")

    if '2022-01-01' <= df_weights.index.max().strftime('%Y-%m-%d') and '2022-12-31' >= df_weights.index.min().strftime('%Y-%m-%d'):
        weights_2022 = df_weights.loc['2022']
        if not weights_2022.empty:
            mean_weights_2022 = weights_2022.mean()
            energia = ['XOM', 'CVX', 'COP', 'SLB']
            peso_energia = mean_weights_2022[energia].sum()
            print(f"Concentración media en Energía durante 2022: {peso_energia:.2%}")


if __name__ == "__main__":
    analyze_weights()
