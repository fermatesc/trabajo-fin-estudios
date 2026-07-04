"""Backtesting walk-forward out-of-sample cableado a la configuración base
(4 qubits, 2 capas cuánticas) y a pesos `.pth` pre-entrenados por semilla.

Compara QGNN, su gemelo clásico, Markowitz y Equal-Weight con costes de
transacción y refit periódico. Guarda las métricas en `results_metrics.json`
y el gráfico comparativo en `plots/backtesting_results.png`.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.dataset_loader import FinancialGraphDataset
from models.qgnn_model import QGNN_Portfolio
from models.classical_gnn_model import ClassicalGNN_Portfolio
from models.markowitz_baseline import MarkowitzOptimizer
from evaluation.metrics import calculate_portfolio_metrics, print_metrics_table

from models.loss_functions import DifferentiableSharpeLoss
import torch.optim as optim
from tqdm import tqdm
from config import set_global_seed, SEED, LEARNING_RATE, IN_CHANNELS, HIDDEN_CHANNELS, N_QUBITS, Q_LAYERS, TRANSACTION_COST_BPS

SEEDS = [42, 43, 44, 45, 46]
REFIT_FREQ = 21
EPOCHS_REFIT = 3


def train_refit(model, dataset, start_idx, end_idx):
    """Reentrena (fine-tuning) `model` sobre las muestras [start_idx, end_idx) del dataset."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DifferentiableSharpeLoss(risk_free_rate=0.0)
    model.train()
    for epoch in range(EPOCHS_REFIT):
        for i in range(start_idx, end_idx):
            data = dataset[i]
            optimizer.zero_grad()
            weights = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(weights, data.y, data.cov)
            loss.backward()
            optimizer.step()
    return model


def evaluate_backtest():
    """Ejecuta el backtesting walk-forward completo y genera resultados y gráfico.

    Carga (o entrena si no existen) los pesos de QGNN y su gemelo clásico por
    semilla, evalúa junto a Markowitz y Equal-Weight en el tramo de test con
    refit periódico y costes de transacción, y promedia entre semillas para
    formar la cartera ensemble de los modelos neuronales.

    Notes:
        El periodo de validación se absorbe en el pre-entrenamiento inicial
        (test_start = train_size), no se usa un tramo de validación aparte.
        La cartera Equal-Weight se rebalancea a diario: los pesos derivan con
        los precios y volver a 1/N tiene coste de transacción, igual que las
        demás estrategias. Además de las métricas sobre la cartera ensemble,
        se reporta la dispersión entre semillas (media ± desviación) del
        Sharpe y del retorno anualizado de QGNN y su gemelo clásico, como
        indicador de significancia.
    """
    print("Iniciando evaluación financiera Walk-Forward Out-Of-Sample...")

    dataset = FinancialGraphDataset(tau=0.5)
    total_samples = len(dataset)
    train_size = int(total_samples * 0.70)
    val_size = int(total_samples * 0.15)
    test_start = train_size
    test_indices = list(range(test_start, total_samples))

    returns_dict = {"QGNN": [], "Classical_GNN": [], "Markowitz": [], "Equal_Weight": []}
    turnover_dict = {"QGNN": [], "Classical_GNN": [], "Markowitz": [], "Equal_Weight": []}

    N = dataset.num_assets
    ew_weights = torch.ones(N) / N

    print(f"Evaluando en {len(test_indices)} días de test cronológicos con Walk-Forward (Refit={REFIT_FREQ} días)...")

    print("Calculando Markowitz y Equal Weight...")
    markowitz = MarkowitzOptimizer()
    w_prev_marko = ew_weights.numpy()
    w_prev_ew = ew_weights.numpy()

    for i in test_indices:
        data = dataset[i]
        idx_date = dataset.valid_dates[i]
        window_returns = dataset.returns.loc[:idx_date].iloc[-60:].values

        w_marko = markowitz.optimize(window_returns)

        y_simple = np.expm1(data.y.numpy())

        turnover_marko = np.abs(w_marko - w_prev_marko).sum()
        ret_marko = np.dot(w_marko, y_simple) - (turnover_marko * TRANSACTION_COST_BPS / 10000.0)

        w_target_ew = ew_weights.numpy()
        turnover_ew = np.abs(w_target_ew - w_prev_ew).sum()
        ret_ew = np.dot(w_target_ew, y_simple) - (turnover_ew * TRANSACTION_COST_BPS / 10000.0)
        w_drift_ew = w_target_ew * (1.0 + y_simple)
        w_prev_ew = w_drift_ew / w_drift_ew.sum()

        returns_dict["Markowitz"].append(ret_marko)
        returns_dict["Equal_Weight"].append(ret_ew)
        turnover_dict["Markowitz"].append(turnover_marko)
        turnover_dict["Equal_Weight"].append(turnover_ew)

        w_drift_marko = w_marko * (1.0 + y_simple)
        w_prev_marko = w_drift_marko / w_drift_marko.sum()

    qgnn_all_rets = []
    class_all_rets = []

    for seed in SEEDS:
        print(f"\n--- Ejecutando Walk-Forward para Semilla {seed} ---")
        set_global_seed(seed)

        qgnn_model = QGNN_Portfolio(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, n_qubits=N_QUBITS, q_layers=Q_LAYERS)
        classical_model = ClassicalGNN_Portfolio(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, latent_dim=4)

        try:
            qgnn_model.load_state_dict(torch.load(f"qgnn_model_seed_{seed}.pth"))
            classical_model.load_state_dict(torch.load(f"classical_gnn_model_seed_{seed}.pth"))
        except FileNotFoundError:
            print("Aviso: No se encontraron pesos iniciales. Se entrenará desde cero en la primera ventana.")
            qgnn_model = train_refit(qgnn_model, dataset, 0, test_start)
            classical_model = train_refit(classical_model, dataset, 0, test_start)

        w_prev_qgnn = ew_weights.numpy()
        w_prev_class = ew_weights.numpy()

        seed_qgnn_rets = []
        seed_class_rets = []
        seed_qgnn_turn = []
        seed_class_turn = []

        current_train_end = test_start

        for idx, i in enumerate(tqdm(test_indices, desc=f"Walk-Forward Seed {seed}")):
            if idx > 0 and idx % REFIT_FREQ == 0:
                qgnn_model = train_refit(qgnn_model, dataset, current_train_end, i)
                classical_model = train_refit(classical_model, dataset, current_train_end, i)
                current_train_end = i

            qgnn_model.eval()
            classical_model.eval()

            data = dataset[i]
            with torch.no_grad():
                w_qgnn = qgnn_model(data.x, data.edge_index, data.edge_attr).numpy()
                w_class = classical_model(data.x, data.edge_index, data.edge_attr).numpy()

            y_simple = np.expm1(data.y.numpy())

            turnover_qgnn = np.abs(w_qgnn - w_prev_qgnn).sum()
            ret_qgnn = np.dot(w_qgnn, y_simple) - (turnover_qgnn * TRANSACTION_COST_BPS / 10000.0)

            turnover_class = np.abs(w_class - w_prev_class).sum()
            ret_class = np.dot(w_class, y_simple) - (turnover_class * TRANSACTION_COST_BPS / 10000.0)

            seed_qgnn_rets.append(ret_qgnn)
            seed_class_rets.append(ret_class)
            seed_qgnn_turn.append(turnover_qgnn)
            seed_class_turn.append(turnover_class)

            w_drift_qgnn = w_qgnn * (1.0 + y_simple)
            w_prev_qgnn = w_drift_qgnn / w_drift_qgnn.sum()

            w_drift_class = w_class * (1.0 + y_simple)
            w_prev_class = w_drift_class / w_drift_class.sum()

        qgnn_all_rets.append(seed_qgnn_rets)
        class_all_rets.append(seed_class_rets)
        turnover_dict["QGNN"].append(float(np.mean(seed_qgnn_turn)))
        turnover_dict["Classical_GNN"].append(float(np.mean(seed_class_turn)))

    returns_dict["QGNN"] = np.mean(qgnn_all_rets, axis=0).tolist()
    returns_dict["Classical_GNN"] = np.mean(class_all_rets, axis=0).tolist()

    metrics_results = {}
    ew_rets = np.array(returns_dict["Equal_Weight"])

    for model_name, rets in returns_dict.items():
        rets_arr = np.array(rets)
        metrics_results[model_name] = calculate_portfolio_metrics(rets_arr, benchmark_returns=ew_rets)
        metrics_results[model_name]["turnover_medio"] = float(np.mean(turnover_dict[model_name]))

    for model_name, all_rets in [("QGNN", qgnn_all_rets), ("Classical_GNN", class_all_rets)]:
        per_seed = [calculate_portfolio_metrics(np.array(r), benchmark_returns=ew_rets) for r in all_rets]
        sharpes = [m["sharpe_ratio"] for m in per_seed]
        rets_an = [m["retorno_anualizado"] for m in per_seed]
        metrics_results[model_name]["sharpe_per_seed"] = sharpes
        metrics_results[model_name]["sharpe_seed_mean"] = float(np.mean(sharpes))
        metrics_results[model_name]["sharpe_seed_std"] = float(np.std(sharpes))
        metrics_results[model_name]["ret_anual_seed_mean"] = float(np.mean(rets_an))
        metrics_results[model_name]["ret_anual_seed_std"] = float(np.std(rets_an))
        
    print("\n--- RESULTADOS DE BACKTESTING ---")
    print_metrics_table(metrics_results)
    
    with open("results_metrics.json", "w") as f:
        json.dump(metrics_results, f, indent=4)
        
    plt.figure(figsize=(12, 6))
    colors = {"QGNN": "darkblue", "Classical_GNN": "orange", "Markowitz": "green", "Equal_Weight": "gray"}
    linestyles = {"QGNN": "-", "Classical_GNN": "-", "Markowitz": "--", "Equal_Weight": ":"}
    linewidths = {"QGNN": 2.5, "Classical_GNN": 1.5, "Markowitz": 1.5, "Equal_Weight": 1.5}
    
    for model_name, rets in returns_dict.items():
        cum_ret = np.cumprod(1 + np.array(rets))
        plt.plot(cum_ret, label=model_name.replace("_", " "), 
                 color=colors[model_name], 
                 linestyle=linestyles[model_name],
                 linewidth=linewidths[model_name])
                 
    plt.title("Backtesting Walk-Forward con Costes de Transacción (Media 5 Semillas)", fontsize=14, weight='bold')
    plt.xlabel("Días de Operación (Test Set)", fontsize=12)
    plt.ylabel("Crecimiento Acumulado", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/backtesting_results.png")
    plt.close()
    
    print("\nGráfico comparativo guardado en 'plots/backtesting_results.png'.")


if __name__ == "__main__":
    evaluate_backtest()
