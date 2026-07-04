"""
evaluate_risk.py — Harness de evaluación centrado en RIESGO (con costes netos).

OBJETIVO:
    Entrenar y backtestear, CON COSTES NETOS de transacción, estrategias
    centradas en la gestión del riesgo de cartera, enfatizando métricas de
    cola: Equal-Weight (referencia), HRP (Hierarchical Risk Parity sobre el
    grafo de correlaciones, determinista, sin entrenamiento) y dos variantes
    entrenadas de QGNN_V2 orientadas a riesgo (mínima varianza y CVaR).

VARIANTES:
    - Equal_Weight : referencia sin modelo (drift diario + coste de
                     rebalanceo a 1/N).
    - HRP          : hrp_weights(data.cov) recalculado cada día de test.
                     Determinista (sin semilla) → una sola pasada.
    - V2_MinVar    : QGNN_V2(temperature=1.5) entrenado con
                     MinVarianceLoss(turnover_lambda=0.0015).
    - V2_CVaR      : QGNN_V2(temperature=1.5) entrenado con
                     CVaRLoss(alpha=0.95, turnover_lambda=0.0015).

DATASET:
    SequenceGraphDataset(data_dir=<nombre>, tau=0.5, lookback=RISK_LOOKBACK)
    — por defecto "data_large" y "data_xl".

SPLITS (idénticos a evaluate_all.py):
    total = len(dataset)
    train_size = int(0.70 * total); val_size = int(0.15 * total)
    train_idx = [i en ds.valid_range() si i < train_size]
    test_idx  = [i en ds.valid_range() si i >= train_size + val_size]
    (el backtest arranca siempre con w_prev = 1/N)

MÉTRICA DE COLA ADICIONAL — CVaR-95 histórico:
    Sobre la serie de retornos NETOS diarios de cada estrategia: se ordenan
    las pérdidas (-rets), se toma el peor 5% (k = max(1, int(0.05*n))) y se
    promedia. Valor positivo = pérdida diaria esperada en la cola.

PARAMETRIZACIÓN (variables de entorno, con defaults conservadores):
    RISK_EPOCHS    (int, default 5)
    RISK_SEEDS     (csv, default "42,43,44")
    RISK_DATASETS  (csv, default "data_large,data_xl")
    RISK_VARIANTS  (csv, default "HRP,V2_MinVar,V2_CVaR")
    RISK_LOOKBACK  (int, default 20)

USO:
    Run por defecto:
        python -m training.evaluate_risk

    Smoke test rápido (PowerShell):
        $env:RISK_EPOCHS=1; $env:RISK_SEEDS="42"; $env:RISK_VARIANTS="HRP,V2_CVaR"
        $env:RISK_DATASETS="data_large"
        .venv\\Scripts\\python.exe -m training.evaluate_risk

SALIDA:
    - Progreso por seed/variante con tiempo de entrenamiento (flush=True).
    - Tabla comparativa final por dataset (Sharpe±std, Sortino, MaxDD,
      CVaR95, Turnover).
    - JSON con guardado incremental en evaluate_risk.json (raíz del repo),
      con un bloque "_setup" con epochs/seeds/datasets/lookback/variantes.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils.sequence_dataset import SequenceGraphDataset
from models.qgnn_v2 import QGNN_V2
from models.hrp_allocator import hrp_weights
from models.loss_functions import MinVarianceLoss, CVaRLoss
from evaluation.metrics import calculate_portfolio_metrics
from config import (
    set_global_seed,
    LEARNING_RATE,
    IN_CHANNELS,
    TRANSACTION_COST_BPS,
)

RISK_EPOCHS = int(os.environ.get("RISK_EPOCHS", "5"))

_seeds_env = os.environ.get("RISK_SEEDS", "42,43,44")
RISK_SEEDS = [int(s.strip()) for s in _seeds_env.split(",") if s.strip()]

_datasets_env = os.environ.get("RISK_DATASETS", "data_large,data_xl")
RISK_DATASETS = [d.strip() for d in _datasets_env.split(",") if d.strip()]

_variants_env = os.environ.get("RISK_VARIANTS", "HRP,V2_MinVar,V2_CVaR")
RISK_VARIANTS = [v.strip() for v in _variants_env.split(",") if v.strip()]

RISK_LOOKBACK = int(os.environ.get("RISK_LOOKBACK", "20"))

TRAINED_VARIANTS = {"V2_MinVar", "V2_CVaR"}

QGNN_V2_KWARGS = dict(
    in_channels=IN_CHANNELS,
    hidden_channels=8,
    n_qubits=4,
    q_layers=2,
    gru_hidden=16,
    temperature=1.5,
    reupload=True,
)

TRANSACTION_COST = TRANSACTION_COST_BPS / 10_000.0

_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "evaluate_risk.json",
)


def historical_cvar95(rets_net):
    """Calcula el CVaR-95 histórico de una serie de retornos netos diarios.

    Ordena las pérdidas (-rets) de mayor a menor y promedia el peor 5%
    (k = max(1, int(0.05*n))).

    Args:
        rets_net: Secuencia de retornos netos diarios de la cartera.

    Returns:
        float: Pérdida diaria esperada en la cola (valor positivo = pérdida).
    """
    losses = -np.asarray(rets_net, dtype=float)
    n = len(losses)
    k = max(1, int(0.05 * n))
    worst = np.sort(losses)[::-1][:k]
    return float(np.mean(worst))


def build_criterion(variant_name):
    """Devuelve el criterio (loss) para una variante entrenada.

    Args:
        variant_name: Nombre de la variante ("V2_MinVar" o "V2_CVaR").

    Returns:
        Instancia del criterio de pérdida correspondiente.

    Raises:
        ValueError: Si `variant_name` no es una variante entrenada conocida.
    """
    if variant_name == "V2_MinVar":
        return MinVarianceLoss(turnover_lambda=0.0015, anchor_lambda=0.0)
    elif variant_name == "V2_CVaR":
        return CVaRLoss(alpha=0.95, turnover_lambda=0.0015, anchor_lambda=0.0)
    else:
        raise ValueError(f"Variante entrenada desconocida: {variant_name}")


def train_variant(model, dataset, train_indices, variant_name, epochs):
    """Entrena `model` con la pérdida de `variant_name` sobre `train_indices`.

    Reinicia `prev_w` al inicio de cada época.

    Args:
        model: Instancia de QGNN_V2 a entrenar.
        dataset: SequenceGraphDataset del que se extraen los grafos de train.
        train_indices: Índices de entrenamiento sobre `dataset`.
        variant_name: "V2_MinVar" o "V2_CVaR", determina la pérdida usada.
        epochs: Número de épocas de entrenamiento.

    Notes:
        MinVarianceLoss expone forward(weights, returns=None, cov_matrix=None,
        prev_weights=None); CVaRLoss expone forward(weights, returns,
        cov_matrix=None, prev_weights=None). Por eso cada variante pasa los
        argumentos posicionales/nombrados de forma distinta.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = build_criterion(variant_name)

    model.train()
    for _epoch in range(epochs):
        prev_w = None

        for idx in train_indices:
            data = dataset[idx]
            optimizer.zero_grad()
            w = model(data.x_seq, data.edge_index, data.edge_attr)

            if variant_name == "V2_MinVar":
                loss = criterion(w, cov_matrix=data.cov, prev_weights=prev_w)
            elif variant_name == "V2_CVaR":
                loss = criterion(w, data.y, data.cov, prev_weights=prev_w)
            else:
                raise ValueError(f"Variante entrenada desconocida: {variant_name}")

            loss.backward()
            optimizer.step()

            prev_w = w.detach()


def backtest_with_costs(model, dataset, test_indices, n_assets):
    """Evalúa el modelo sobre test_indices aplicando costes de transacción.

    `w_prev` arranca en Equal-Weight (1/N).

    Args:
        model: Modelo QGNN_V2 ya entrenado.
        dataset: SequenceGraphDataset del que se extraen los grafos de test.
        test_indices: Índices de test sobre `dataset`.
        n_assets: Número de activos de la cartera.

    Returns:
        tuple[list[float], list[float]]: Retornos netos diarios y turnovers
        diarios de la estrategia.
    """
    w_prev = np.ones(n_assets) / n_assets

    rets_net = []
    turnovers = []

    model.eval()
    with torch.no_grad():
        for idx in test_indices:
            data = dataset[idx]
            y_simple = np.expm1(data.y.numpy())
            w = model(data.x_seq, data.edge_index, data.edge_attr).numpy()
            turnover = np.abs(w - w_prev).sum()
            ret_net = np.dot(w, y_simple) - turnover * TRANSACTION_COST
            rets_net.append(float(ret_net))
            turnovers.append(float(turnover))
            w_prev = w

    return rets_net, turnovers


def compute_ew_with_costs(dataset, test_indices, n_assets):
    """Calcula la serie de retornos netos de la cartera Equal-Weight.

    Modela la deriva de los pesos entre rebalanceos diarios y el coste de
    volver a 1/N en cada rebalanceo. Réplica de la lógica usada en
    evaluate_all.py.

    Args:
        dataset: SequenceGraphDataset del que se extraen los grafos de test.
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

    for idx in test_indices:
        data = dataset[idx]
        y_simple = np.expm1(data.y.numpy())

        turnover_ew = np.abs(w_target - w_prev_ew).sum()
        ret_ew = np.dot(w_target, y_simple) - turnover_ew * TRANSACTION_COST

        w_drift = w_target * (1.0 + y_simple)
        w_prev_ew = w_drift / w_drift.sum()

        rets_net.append(float(ret_ew))
        turnovers.append(float(turnover_ew))

    return rets_net, turnovers


def compute_hrp_with_costs(dataset, test_indices, n_assets):
    """Calcula la serie de retornos netos de la cartera HRP.

    Para cada día de test se recalculan los pesos HRP a partir de la matriz
    de covarianza del día (`data.cov`), sin entrenamiento ni semilla
    (estrategia determinista). `w_prev` arranca en Equal-Weight (1/N).

    Args:
        dataset: SequenceGraphDataset del que se extraen los grafos de test.
        test_indices: Índices de test sobre `dataset`.
        n_assets: Número de activos de la cartera.

    Returns:
        tuple[list[float], list[float]]: Retornos netos diarios y turnovers
        diarios de la estrategia HRP.
    """
    w_prev = np.ones(n_assets) / n_assets

    rets_net = []
    turnovers = []

    for idx in test_indices:
        data = dataset[idx]
        y_simple = np.expm1(data.y.numpy())
        w = hrp_weights(data.cov.numpy())
        turnover = np.abs(w - w_prev).sum()
        ret_net = np.dot(w, y_simple) - turnover * TRANSACTION_COST
        rets_net.append(float(ret_net))
        turnovers.append(float(turnover))
        w_prev = w

    return rets_net, turnovers


def build_splits(ds):
    """Construye los índices de train y test sobre un SequenceGraphDataset.

    Args:
        ds: SequenceGraphDataset ya construido.

    Returns:
        tuple: (train_indices, test_indices, total, train_size, val_size),
        donde train_size = int(0.70*total), val_size = int(0.15*total),
        train_indices son los índices válidos con i < train_size y
        test_indices los índices válidos con i >= train_size + val_size.
    """
    total = len(ds)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)

    valid = list(ds.valid_range())
    train_indices = [i for i in valid if i < train_size]
    test_indices = [i for i in valid if i >= train_size + val_size]

    return train_indices, test_indices, total, train_size, val_size


def metrics_with_cvar(rets_net):
    """Calcula las métricas de cartera estándar más el CVaR-95.

    Args:
        rets_net: Secuencia de retornos netos diarios de la cartera.

    Returns:
        dict: Métricas de `calculate_portfolio_metrics` más la clave "cvar95".
    """
    m = calculate_portfolio_metrics(np.array(rets_net))
    m["cvar95"] = historical_cvar95(rets_net)
    return m


def save_incremental(all_results, setup):
    """Escribe `all_results` y `setup` en el JSON de salida (_OUTPUT_PATH).

    Args:
        all_results: Diccionario anidado con los resultados por dataset y
            variante acumulados hasta el momento.
        setup: Diccionario con la configuración del experimento (epochs,
            seeds, datasets, variantes, lookback, coste de transacción).
    """
    payload = dict(all_results)
    payload["_setup"] = setup
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=4)


def main():
    """Ejecuta el harness de riesgo (EW / HRP / QGNN_V2) sobre todos los datasets configurados."""
    print("=" * 70, flush=True)
    print("  evaluate_risk.py — Harness de RIESGO (EW / HRP / QGNN_V2)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Epocas: {RISK_EPOCHS} | Semillas: {RISK_SEEDS}", flush=True)
    print(f"  Datasets: {RISK_DATASETS} | Variantes: {RISK_VARIANTS}", flush=True)
    print(f"  Lookback: {RISK_LOOKBACK} | TRANSACTION_COST_BPS: {TRANSACTION_COST_BPS}", flush=True)
    print("=" * 70, flush=True)

    setup = {
        "epochs": RISK_EPOCHS,
        "seeds": RISK_SEEDS,
        "datasets": RISK_DATASETS,
        "variants": RISK_VARIANTS,
        "lookback": RISK_LOOKBACK,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
    }

    all_results = {}

    for ds_name in RISK_DATASETS:
        print(f"\n{'#' * 70}", flush=True)
        print(f"  DATASET: {ds_name}", flush=True)
        print(f"{'#' * 70}", flush=True)

        ds = SequenceGraphDataset(data_dir=ds_name, tau=0.5, lookback=RISK_LOOKBACK)
        n_assets = ds.num_assets

        train_indices, test_indices, total, train_size, val_size = build_splits(ds)

        print(
            f"  Total grafos: {total} | train_idx: {len(train_indices)} | "
            f"test_idx: {len(test_indices)} | activos: {n_assets}",
            flush=True,
        )

        ds_results = {}

        print("\n--- Calculando Equal-Weight ---", flush=True)
        ew_rets, ew_turns = compute_ew_with_costs(ds, test_indices, n_assets)
        ew_metrics = metrics_with_cvar(ew_rets)
        ew_turnover_medio = float(np.mean(ew_turns))
        print(
            f"  EW Sharpe_neto={ew_metrics['sharpe_ratio']:.3f} | "
            f"Sortino={ew_metrics['sortino_ratio']:.3f} | "
            f"MaxDD={ew_metrics['max_drawdown']:.2%} | "
            f"CVaR95={ew_metrics['cvar95']:.4%} | "
            f"Turnover_medio={ew_turnover_medio:.4f}",
            flush=True,
        )
        ds_results["Equal_Weight"] = {
            "sharpe_neto": float(ew_metrics["sharpe_ratio"]),
            "sortino_neto": float(ew_metrics["sortino_ratio"]),
            "max_drawdown": float(ew_metrics["max_drawdown"]),
            "cvar95": float(ew_metrics["cvar95"]),
            "retorno_neto_anualizado": float(ew_metrics["retorno_anualizado"]),
            "turnover_medio_diario": ew_turnover_medio,
        }
        all_results[ds_name] = ds_results
        save_incremental(all_results, setup)

        if "HRP" in RISK_VARIANTS:
            print("\n--- Calculando HRP ---", flush=True)
            t0 = time.time()
            hrp_rets, hrp_turns = compute_hrp_with_costs(ds, test_indices, n_assets)
            elapsed = time.time() - t0
            hrp_metrics = metrics_with_cvar(hrp_rets)
            hrp_turnover_medio = float(np.mean(hrp_turns))
            print(
                f"  HRP Sharpe_neto={hrp_metrics['sharpe_ratio']:.3f} | "
                f"Sortino={hrp_metrics['sortino_ratio']:.3f} | "
                f"MaxDD={hrp_metrics['max_drawdown']:.2%} | "
                f"CVaR95={hrp_metrics['cvar95']:.4%} | "
                f"Turnover_medio={hrp_turnover_medio:.4f} | "
                f"Tiempo={elapsed:.2f}s",
                flush=True,
            )
            ds_results["HRP"] = {
                "sharpe_neto": float(hrp_metrics["sharpe_ratio"]),
                "sortino_neto": float(hrp_metrics["sortino_ratio"]),
                "max_drawdown": float(hrp_metrics["max_drawdown"]),
                "cvar95": float(hrp_metrics["cvar95"]),
                "retorno_neto_anualizado": float(hrp_metrics["retorno_anualizado"]),
                "turnover_medio_diario": hrp_turnover_medio,
            }
            all_results[ds_name] = ds_results
            save_incremental(all_results, setup)

        for variant_name in RISK_VARIANTS:
            if variant_name not in TRAINED_VARIANTS:
                continue

            print(f"\n{'=' * 70}", flush=True)
            print(f"  Variante: {variant_name}  (dataset={ds_name})", flush=True)
            print("=" * 70, flush=True)

            sharpes, sortinos, maxdds, cvar95s, rets_an, turnovers_medios = (
                [], [], [], [], [], [],
            )

            for seed in RISK_SEEDS:
                set_global_seed(seed)

                model = QGNN_V2(**QGNN_V2_KWARGS)

                t0 = time.time()
                train_variant(
                    model, ds, train_indices, variant_name,
                    epochs=RISK_EPOCHS,
                )
                elapsed = time.time() - t0
                print(
                    f"  [{variant_name} seed={seed}] Entrenamiento: {elapsed/60:.2f} min",
                    flush=True,
                )

                rets_net, turns = backtest_with_costs(model, ds, test_indices, n_assets)
                m = metrics_with_cvar(rets_net)

                sharpes.append(m["sharpe_ratio"])
                sortinos.append(m["sortino_ratio"])
                maxdds.append(m["max_drawdown"])
                cvar95s.append(m["cvar95"])
                rets_an.append(m["retorno_anualizado"])
                turnovers_medios.append(float(np.mean(turns)))

                print(
                    f"  {variant_name} seed={seed}: "
                    f"Sharpe_neto={m['sharpe_ratio']:.3f}  "
                    f"Sortino={m['sortino_ratio']:.3f}  "
                    f"MaxDD={m['max_drawdown']:.2%}  "
                    f"CVaR95={m['cvar95']:.4%}  "
                    f"Turnover_medio={np.mean(turns):.4f}",
                    flush=True,
                )

                ds_results[f"{variant_name}__en_progreso"] = {
                    "seeds_hechas": RISK_SEEDS[: len(sharpes)],
                    "sharpe_per_seed": [float(s) for s in sharpes],
                    "sortino_per_seed": [float(s) for s in sortinos],
                    "max_drawdown_per_seed": [float(s) for s in maxdds],
                    "cvar95_per_seed": [float(s) for s in cvar95s],
                    "turnover_per_seed": [float(t) for t in turnovers_medios],
                }
                all_results[ds_name] = ds_results
                save_incremental(all_results, setup)

            ds_results[variant_name] = {
                "sharpe_per_seed": [float(s) for s in sharpes],
                "sharpe_mean": float(np.mean(sharpes)),
                "sharpe_std": float(np.std(sharpes)),
                "sortino_per_seed": [float(s) for s in sortinos],
                "sortino_mean": float(np.mean(sortinos)),
                "sortino_std": float(np.std(sortinos)),
                "max_drawdown_per_seed": [float(s) for s in maxdds],
                "max_drawdown_mean": float(np.mean(maxdds)),
                "max_drawdown_std": float(np.std(maxdds)),
                "cvar95_per_seed": [float(s) for s in cvar95s],
                "cvar95_mean": float(np.mean(cvar95s)),
                "cvar95_std": float(np.std(cvar95s)),
                "turnover_per_seed": [float(t) for t in turnovers_medios],
                "turnover_mean": float(np.mean(turnovers_medios)),
                "turnover_std": float(np.std(turnovers_medios)),
                "retorno_neto_anualizado_medio": float(np.mean(rets_an)),
            }
            ds_results.pop(f"{variant_name}__en_progreso", None)
            all_results[ds_name] = ds_results
            save_incremental(all_results, setup)

            print(
                f"\n==> {variant_name}: Sharpe={ds_results[variant_name]['sharpe_mean']:.3f} "
                f"+/- {ds_results[variant_name]['sharpe_std']:.3f}  "
                f"Sortino={ds_results[variant_name]['sortino_mean']:.3f}  "
                f"MaxDD={ds_results[variant_name]['max_drawdown_mean']:.2%}  "
                f"CVaR95={ds_results[variant_name]['cvar95_mean']:.4%}  "
                f"Turnover={ds_results[variant_name]['turnover_mean']:.4f}\n",
                flush=True,
            )

        print("\n" + "=" * 70, flush=True)
        print(f"  TABLA COMPARATIVA (RIESGO) — dataset={ds_name}", flush=True)
        print(
            f"  Setup: {n_assets} activos | {RISK_EPOCHS} epocas | "
            f"{len(RISK_SEEDS)} semillas | lookback={RISK_LOOKBACK}",
            flush=True,
        )
        print("=" * 70, flush=True)
        header = (
            f"{'Variante':<18} {'Sharpe':>10} {'+-std':>8} "
            f"{'Sortino':>10} {'MaxDD':>10} {'CVaR95':>10} {'Turnover':>10}"
        )
        print(header, flush=True)
        print("-" * len(header), flush=True)

        for variant_name in RISK_VARIANTS:
            if variant_name not in TRAINED_VARIANTS:
                continue
            r = ds_results[variant_name]
            print(
                f"{variant_name:<18} "
                f"{r['sharpe_mean']:>10.3f} "
                f"{r['sharpe_std']:>8.3f} "
                f"{r['sortino_mean']:>10.3f} "
                f"{r['max_drawdown_mean']:>10.2%} "
                f"{r['cvar95_mean']:>10.4%} "
                f"{r['turnover_mean']:>10.4f}",
                flush=True,
            )

        if "HRP" in RISK_VARIANTS and "HRP" in ds_results:
            hrp = ds_results["HRP"]
            print(
                f"{'HRP':<18} "
                f"{hrp['sharpe_neto']:>10.3f} "
                f"{'--':>8} "
                f"{hrp['sortino_neto']:>10.3f} "
                f"{hrp['max_drawdown']:>10.2%} "
                f"{hrp['cvar95']:>10.4%} "
                f"{hrp['turnover_medio_diario']:>10.4f}",
                flush=True,
            )

        ew = ds_results["Equal_Weight"]
        print(
            f"{'Equal_Weight':<18} "
            f"{ew['sharpe_neto']:>10.3f} "
            f"{'--':>8} "
            f"{ew['sortino_neto']:>10.3f} "
            f"{ew['max_drawdown']:>10.2%} "
            f"{ew['cvar95']:>10.4%} "
            f"{ew['turnover_medio_diario']:>10.4f}",
            flush=True,
        )
        print("=" * 70, flush=True)

    print(f"\nResultados guardados en: {_OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
