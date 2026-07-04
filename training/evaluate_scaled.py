"""
evaluate_scaled.py — Experimento QGNN escalado con dataset completo (62 activos, ~1198 grafos).

OBJETIVO:
    Reproducir el experimento de evaluate_fix.py sobre el dataset ampliado a 62 activos
    del S&P 500 para verificar si la ventaja de la variante FIXED (TurnoverAwareSharpeLoss)
    se mantiene a escala real. Se comparan Sharpe neto y turnover medio contra Equal-Weight.

VARIANTES:
    - BASELINE: QGNN(temperature=5.0) + DifferentiableSharpeLoss()
    - FIXED:    QGNN(temperature=1.5) + TurnoverAwareSharpeLoss(turnover_lambda=0.0015)
                con prev_weights encadenado y detached entre pasos de entrenamiento.

BACKTEST:
    Ambas variantes se evalúan con costes de transacción reales (TRANSACTION_COST_BPS=10
    desde config.py). El Sharpe reportado es el Sharpe NETO (sobre retornos después de
    descontar costes). Equal-Weight sirve de referencia con su coste real de rebalanceo.

DATASET:
    FinancialGraphDataset(data_dir="data_large", tau=0.5)
    — 62 activos S&P 500, ~1198 grafos dinámicos.

ESTRUCTURA:
    - Splits: 70% train / 15% val / 15% test
    - Semillas: configurable vía SCALED_SEEDS (env var, default: "42,43,44")
    - Épocas:   configurable vía SCALED_EPOCHS (env var, default: 30)
    - Hiperparámetros QGNN: n_qubits=4, q_layers=2, hidden=8

PARAMETRIZACIÓN (para pruebas rápidas sin editar el fichero):
    SCALED_EPOCHS=1 SCALED_SEEDS=42 python -m training.evaluate_scaled
    — Reduce a 1 época y 1 semilla para smoke test.

    Para el run completo (defaults):
    python -m training.evaluate_scaled

JUSTIFICACIÓN DE 3 SEMILLAS POR DEFECTO:
    Con 62 activos × 30 épocas × 2 variantes el tiempo de cómputo es significativo.
    3 semillas es suficiente para estimar media ± std de Sharpe con varianza razonable.
    Si el benchmark de tiempo indica que 5 semillas caben en <2h, subir SEEDS a
    [42, 43, 44, 45, 46] manualmente o vía SCALED_SEEDS=42,43,44,45,46.
"""

import json
import sys
import os
import time
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

EXP_EPOCHS = int(os.environ.get("SCALED_EPOCHS", "30"))

_seeds_env = os.environ.get("SCALED_SEEDS", "42,43,44")
SEEDS = [int(s.strip()) for s in _seeds_env.split(",") if s.strip()]

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
    """Ejecuta el experimento QGNN escalado (BASELINE vs FIXED) sobre data_large."""
    print("=" * 65, flush=True)
    print("  evaluate_scaled.py — QGNN escalado (data_large, 62 activos)", flush=True)
    print("=" * 65, flush=True)
    print(f"  Épocas: {EXP_EPOCHS}  |  Semillas: {SEEDS}", flush=True)
    print("=" * 65, flush=True)

    print("\nCargando dataset data_large...", flush=True)
    dataset = FinancialGraphDataset(data_dir="data_large", tau=0.5)
    total = len(dataset)
    n_assets = dataset.num_assets
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)
    test_indices = list(range(train_size + val_size, total))

    print(f"\nDataset: {total} dias | train={train_size} | val={val_size} | test={len(test_indices)}", flush=True)
    print(f"Activos: {n_assets} | TRANSACTION_COST_BPS={TRANSACTION_COST_BPS}", flush=True)
    print(f"Epocas: {EXP_EPOCHS} | Semillas: {SEEDS}", flush=True)

    print("\n--- Calculando Equal-Weight ---", flush=True)
    ew_rets, ew_turns = compute_ew_with_costs(dataset, test_indices, n_assets)
    ew_metrics = calculate_portfolio_metrics(np.array(ew_rets))
    ew_turnover_medio = float(np.mean(ew_turns))
    print(f"EW Sharpe neto: {ew_metrics['sharpe_ratio']:.3f} | Turnover medio: {ew_turnover_medio:.4f}", flush=True)

    results = {}

    _OUTPUT_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluate_scaled.json",
    )

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

            t0 = time.time()
            variant["train_fn"](model, dataset, train_size)
            elapsed = time.time() - t0
            print(f"  [{vname} seed={seed}] Entrenamiento: {elapsed/60:.1f} min", flush=True)

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

            _partial = dict(results)
            _partial[f"{vname}__en_progreso"] = {
                "seeds_hechas": SEEDS[:len(sharpes)],
                "sharpe_per_seed": [float(s) for s in sharpes],
                "turnover_per_seed": [float(t) for t in all_turnover_medios],
            }
            _partial["Equal_Weight"] = {
                "sharpe_neto": float(ew_metrics["sharpe_ratio"]),
                "turnover_medio_diario": ew_turnover_medio,
                "retorno_neto_anualizado": float(ew_metrics["retorno_anualizado"]),
            }
            with open(_OUTPUT_PATH, "w") as _f:
                json.dump(_partial, _f, indent=4)

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

    results["_setup"] = {
        "n_activos": n_assets,
        "epocas": EXP_EPOCHS,
        "semillas": SEEDS,
        "n_semillas": len(SEEDS),
        "total_grafos": total,
        "train_size": train_size,
        "test_size": len(test_indices),
        "dataset": "data_large",
        "transaction_cost_bps": TRANSACTION_COST_BPS,
    }

    print("\n" + "=" * 65, flush=True)
    print("  TABLA COMPARATIVA FINAL (costes netos incluidos)", flush=True)
    print(f"  Setup: {n_assets} activos | {EXP_EPOCHS} epocas | {len(SEEDS)} semillas", flush=True)
    print("=" * 65, flush=True)
    header = f"{'Variante':<16} {'Sharpe_neto':>12} {'+-std':>8} {'Turnover':>10} {'Ret_Anual':>11}"
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
        f"{'--':>8} "
        f"{ew['turnover_medio_diario']:>10.4f} "
        f"{ew['retorno_neto_anualizado']:>10.2%}",
        flush=True,
    )
    print("=" * 65, flush=True)

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "evaluate_scaled.json",
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResultados guardados en: {output_path}", flush=True)


if __name__ == "__main__":
    main()
