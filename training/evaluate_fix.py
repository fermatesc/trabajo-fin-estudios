"""
evaluate_fix.py — Validación del "fix" de QGNN con costes de transacción netos.

OBJETIVO:
    Verificar si la variante FIXED (temperatura=1.5 + TurnoverAwareSharpeLoss) reduce
    el turnover (objetivo: de ~0.55 hacia <0.2) y si su Sharpe NETO medio sube respecto
    a BASELINE y se acerca o supera al de Equal-Weight (~0.82).

VARIANTES:
    - BASELINE: QGNN(temperature=5.0) + DifferentiableSharpeLoss()
    - FIXED:    QGNN(temperature=1.5) + TurnoverAwareSharpeLoss(turnover_lambda=0.0015)

BACKTEST:
    Ambas variantes se evalúan con costes de transacción reales (TRANSACTION_COST_BPS=10
    desde config.py). El Sharpe reportado es el Sharpe NETO (sobre retornos después de
    descontar costes). Equal-Weight sirve de referencia con su coste real de rebalanceo.

ESTRUCTURA:
    - Dataset: FinancialGraphDataset(tau=0.5)
    - Splits: 70% train / 15% val / 15% test (≈181 días de test)
    - Semillas: [42, 43, 44, 45, 46] — se promedia la varianza entre semillas
    - Hiperparámetros QGNN: n_qubits=4, q_layers=2, hidden=8
    - Épocas por semilla: 5
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
from models.loss_functions import DifferentiableSharpeLoss, TurnoverAwareSharpeLoss
from evaluation.metrics import calculate_portfolio_metrics
from config import (
    set_global_seed,
    LEARNING_RATE,
    IN_CHANNELS,
    TRANSACTION_COST_BPS,
)

EXP_EPOCHS = 5
SEEDS = [42, 43, 44, 45, 46]

QGNN_KWARGS = dict(
    in_channels=IN_CHANNELS,
    hidden_channels=8,
    n_qubits=4,
    q_layers=2,
)

TRANSACTION_COST = TRANSACTION_COST_BPS / 10_000.0


def train_baseline(model, dataset, train_size):
    """Entrena la variante BASELINE sobre range(train_size).

    Usa DifferentiableSharpeLoss, sin encadenar pesos previos entre pasos.

    Args:
        model: Modelo QGNN_Portfolio a entrenar.
        dataset: FinancialGraphDataset del que se extraen los grafos.
        train_size: Número de días de entrenamiento (recorre range(train_size)).
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DifferentiableSharpeLoss()

    model.train()
    for _epoch in range(EXP_EPOCHS):
        for i in range(train_size):
            data = dataset[i]
            optimizer.zero_grad()
            weights = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(weights, data.y, data.cov)
            loss.backward()
            optimizer.step()


def train_fixed(model, dataset, train_size):
    """Entrena la variante FIXED sobre range(train_size).

    Usa TurnoverAwareSharpeLoss, encadenando los pesos del paso anterior
    (`prev_weights`) para penalizar el turnover.

    Args:
        model: Modelo QGNN_Portfolio a entrenar.
        dataset: FinancialGraphDataset del que se extraen los grafos.
        train_size: Número de días de entrenamiento (recorre range(train_size)).

    Notes:
        `prev_w` se guarda con `.detach()` para que el gradiente no se
        propague a través del histórico de pasos anteriores.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = TurnoverAwareSharpeLoss(turnover_lambda=0.0015)

    model.train()
    for _epoch in range(EXP_EPOCHS):
        prev_w = None
        for i in range(train_size):
            data = dataset[i]
            optimizer.zero_grad()
            w = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(w, data.y, data.cov, prev_weights=prev_w)
            loss.backward()
            optimizer.step()
            prev_w = w.detach()


def backtest_with_costs(model, dataset, test_indices, n_assets):
    """Evalúa el modelo sobre test_indices aplicando costes de transacción.

    `w_prev` arranca en Equal-Weight (1/N) para ser consistente con el
    estado real de la cartera antes del período de test.

    Args:
        model: Modelo QGNN_Portfolio ya entrenado.
        dataset: FinancialGraphDataset del que se extraen los grafos de test.
        test_indices: Índices de test sobre `dataset`.
        n_assets: Número de activos de la cartera.

    Returns:
        tuple[list[float], list[float]]: Retorno neto diario y turnover
        absoluto diario.
    """
    w_prev = np.ones(n_assets) / n_assets

    rets_net = []
    turnovers = []

    model.eval()
    with torch.no_grad():
        for i in test_indices:
            data = dataset[i]
            y_simple = np.expm1(data.y.numpy())
            w = model(data.x, data.edge_index, data.edge_attr).numpy()
            turnover = np.abs(w - w_prev).sum()
            ret_net = np.dot(w, y_simple) - turnover * TRANSACTION_COST
            rets_net.append(float(ret_net))
            turnovers.append(float(turnover))
            w_prev = w

    return rets_net, turnovers


def compute_ew_with_costs(dataset, test_indices, n_assets):
    """Calcula la serie de retornos netos de la cartera Equal-Weight.

    Modela la deriva de los pesos entre rebalanceos diarios y el coste de
    volver a 1/N en cada rebalanceo. Réplica la lógica de
    evaluation/backtesting.py (líneas ~75-79).

    Args:
        dataset: FinancialGraphDataset del que se extraen los grafos de test.
        test_indices: Índices de test sobre `dataset`.
        n_assets: Número de activos de la cartera.

    Returns:
        tuple[list[float], list[float]]: Retornos netos diarios y turnovers
        diarios de la estrategia Equal-Weight.
    """
    w_target = np.ones(n_assets) / n_assets
    w_prev_ew = w_target.copy()

    rets_net = []
    turnovers = []

    for i in test_indices:
        data = dataset[i]
        y_simple = np.expm1(data.y.numpy())

        turnover_ew = np.abs(w_target - w_prev_ew).sum()
        ret_ew = np.dot(w_target, y_simple) - turnover_ew * TRANSACTION_COST

        w_drift = w_target * (1.0 + y_simple)
        w_prev_ew = w_drift / w_drift.sum()

        rets_net.append(float(ret_ew))
        turnovers.append(float(turnover_ew))

    return rets_net, turnovers


def main():
    """Ejecuta la validación del fix de QGNN (BASELINE vs FIXED) con costes netos."""
    print("=" * 65, flush=True)
    print("  evaluate_fix.py — Validación fix QGNN con costes netos", flush=True)
    print("=" * 65, flush=True)

    dataset = FinancialGraphDataset(tau=0.5)
    total = len(dataset)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)
    test_indices = list(range(train_size + val_size, total))
    n_assets = dataset.num_assets

    print(f"\nDataset: {total} dias | train={train_size} | val={val_size} | test={len(test_indices)}", flush=True)
    print(f"Activos: {n_assets} | TRANSACTION_COST_BPS={TRANSACTION_COST_BPS}", flush=True)

    print("\n--- Calculando Equal-Weight ---", flush=True)
    ew_rets, ew_turns = compute_ew_with_costs(dataset, test_indices, n_assets)
    ew_metrics = calculate_portfolio_metrics(np.array(ew_rets))
    ew_turnover_medio = float(np.mean(ew_turns))
    print(f"EW Sharpe neto: {ew_metrics['sharpe_ratio']:.3f} | Turnover medio: {ew_turnover_medio:.4f}", flush=True)

    results = {}

    variants = [
        {
            "name": "BASELINE",
            "temperature": 5.0,
            "train_fn": train_baseline,
        },
        {
            "name": "FIXED",
            "temperature": 1.5,
            "train_fn": train_fixed,
        },
    ]

    for variant in variants:
        vname = variant["name"]
        print(f"\n{'=' * 65}", flush=True)
        print(f"  Variante: {vname}  (temperature={variant['temperature']})", flush=True)
        print("=" * 65, flush=True)

        sharpes = []
        rets_an = []
        all_turnover_medios = []

        for seed in SEEDS:
            set_global_seed(seed)

            model = QGNN_Portfolio(
                temperature=variant["temperature"],
                **QGNN_KWARGS,
            )

            variant["train_fn"](model, dataset, train_size)

            rets_net, turnovers = backtest_with_costs(model, dataset, test_indices, n_assets)

            m = calculate_portfolio_metrics(np.array(rets_net))
            sharpes.append(m["sharpe_ratio"])
            rets_an.append(m["retorno_anualizado"])
            all_turnover_medios.append(float(np.mean(turnovers)))

            print(
                f"  {vname} seed={seed}: "
                f"Sharpe_neto={m['sharpe_ratio']:.3f}  "
                f"Ret_neto_anual={m['retorno_anualizado']:.2%}  "
                f"Turnover_medio={np.mean(turnovers):.4f}",
                flush=True,
            )

        results[vname] = {
            "sharpe_per_seed": [float(s) for s in sharpes],
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_min": float(np.min(sharpes)),
            "sharpe_max": float(np.max(sharpes)),
            "turnover_medio_diario": float(np.mean(all_turnover_medios)),
            "retorno_neto_anualizado_medio": float(np.mean(rets_an)),
            "temperature": variant["temperature"],
        }

        print(
            f"\n==> {vname}: Sharpe_neto={results[vname]['sharpe_mean']:.3f} "
            f"+/- {results[vname]['sharpe_std']:.3f} "
            f"[{results[vname]['sharpe_min']:.3f}, {results[vname]['sharpe_max']:.3f}]  "
            f"Turnover_medio={results[vname]['turnover_medio_diario']:.4f}  "
            f"Ret_anual={results[vname]['retorno_neto_anualizado_medio']:.2%}\n",
            flush=True,
        )

    results["Equal_Weight"] = {
        "sharpe_neto": float(ew_metrics["sharpe_ratio"]),
        "turnover_medio_diario": ew_turnover_medio,
        "retorno_neto_anualizado": float(ew_metrics["retorno_anualizado"]),
    }

    print("\n" + "=" * 65, flush=True)
    print("  TABLA COMPARATIVA FINAL (costes netos incluidos)", flush=True)
    print("=" * 65, flush=True)
    header = f"{'Variante':<16} {'Sharpe_neto':>12} {'±std':>8} {'Turnover':>10} {'Ret_Anual':>11}"
    print(header, flush=True)
    print("-" * 60, flush=True)

    for vname in ["BASELINE", "FIXED"]:
        r = results[vname]
        print(
            f"{vname:<16} "
            f"{r['sharpe_mean']:>12.3f} "
            f"{r['sharpe_std']:>8.3f} "
            f"{r['turnover_medio_diario']:>10.4f} "
            f"{r['retorno_neto_anualizado_medio']:>10.2%}",
            flush=True,
        )

    ew = results["Equal_Weight"]
    print(
        f"{'Equal_Weight':<16} "
        f"{ew['sharpe_neto']:>12.3f} "
        f"{'—':>8} "
        f"{ew['turnover_medio_diario']:>10.4f} "
        f"{ew['retorno_neto_anualizado']:>10.2%}",
        flush=True,
    )
    print("=" * 65, flush=True)

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluate_fix.json",
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados guardados en: {output_path}", flush=True)


if __name__ == "__main__":
    main()
