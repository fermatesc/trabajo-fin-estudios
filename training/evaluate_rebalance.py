"""
evaluate_rebalance.py — Barrido de FRECUENCIA DE REBALANCEO (con costes netos).

MOTIVACIÓN:
    El hallazgo central de la línea de riesgo es que el coste de transacción
    (turnover x 10 bps) es lo que destruye el Sharpe neto de las estrategias
    activas (HRP rota ~0.21/día → ~5% anual de drag). Rebalancear con MENOS
    frecuencia (semanal/mensual) es la palanca natural y realista para
    controlar ese coste: ninguna gestora institucional rebalancea a diario.

    Este harness mide, para cada estrategia, el Sharpe/Sortino/MaxDD/CVaR95
    y el turnover medio en función de la frecuencia de rebalanceo
    (1=diario, 5=semanal, 21=mensual).

MECÁNICA REALISTA DE REBALANCEO (clave):
    Entre rebalanceos NO se opera: los pesos se MANTIENEN y DERIVAN con el
    mercado (w_{t+1} = w_t·(1+r_t) / Σ). Solo el día de rebalanceo se calculan
    los pesos objetivo y se paga el coste = |w_objetivo − w_derivado|·bps.
    El backtest siempre arranca con w = 1/N.

    Nota metodológica: esta contabilidad "con deriva" es más realista que la
    de evaluate_risk.py (que comparaba contra los pesos objetivo del día
    anterior, sin deriva). Por eso el caso freq=1 aquí puede diferir
    marginalmente del de evaluate_risk.json; ambos comparten el veredicto.

ESTRATEGIAS:
    - Equal_Weight : objetivo 1/N en cada rebalanceo.
    - HRP          : hrp_weights(data.cov) recalculado en cada rebalanceo
                     (determinista, sin semilla).
    - V2_MinVar    : QGNN_V2 entrenado con MinVarianceLoss.
    - V2_CVaR      : QGNN_V2 entrenado con CVaRLoss(alpha=0.95).
    Las variantes entrenadas se entrenan UNA vez por semilla (entrenamiento
    diario, agnóstico a la frecuencia) y se backtestean a las 3 frecuencias.

PARAMETRIZACIÓN (variables de entorno):
    REBAL_EPOCHS     (int, default 5)
    REBAL_SEEDS      (csv, default "42,43,44")
    REBAL_DATASETS   (csv, default "data_large,data_xl")
    REBAL_STRATEGIES (csv, default "Equal_Weight,HRP,V2_MinVar,V2_CVaR")
    REBAL_FREQS      (csv, default "1,5,21")        # 1=diario, 5=semanal, 21=mensual
    REBAL_LOOKBACK   (int, default 20)
    REBAL_BLENDS     (csv, default "") — mezclas convexas EW(alpha) + HRP(1-alpha);
                     vacío por defecto (off). Ej.: REBAL_BLENDS="0.25,0.5,0.75"
                     genera las estrategias Blend_EW25 / Blend_EW50 / Blend_EW75.

USO:
    Run por defecto:
        .venv\\Scripts\\python.exe -m training.evaluate_rebalance

    Smoke determinista (rápido, sin entrenar):
        $env:REBAL_STRATEGIES="Equal_Weight,HRP"; $env:REBAL_DATASETS="data_large"
        .venv\\Scripts\\python.exe -m training.evaluate_rebalance

SALIDA:
    evaluate_rebalance.json (raíz del repo), estructura:
        results[dataset]["freq_<F>"][estrategia] = { métricas }
    + bloque "_setup". Guardado incremental tras cada estrategia/semilla.
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
from evaluation.metrics import calculate_portfolio_metrics
from config import (
    set_global_seed,
    LEARNING_RATE,
    IN_CHANNELS,
    TRANSACTION_COST_BPS,
)
from training.evaluate_risk import (
    train_variant,
    build_splits,
    historical_cvar95,
)

REBAL_EPOCHS = int(os.environ.get("REBAL_EPOCHS", "5"))

_seeds_env = os.environ.get("REBAL_SEEDS", "42,43,44")
REBAL_SEEDS = [int(s.strip()) for s in _seeds_env.split(",") if s.strip()]

_ds_env = os.environ.get("REBAL_DATASETS", "data_large,data_xl")
REBAL_DATASETS = [d.strip() for d in _ds_env.split(",") if d.strip()]

_strat_env = os.environ.get("REBAL_STRATEGIES", "Equal_Weight,HRP,V2_MinVar,V2_CVaR")
REBAL_STRATEGIES = [s.strip() for s in _strat_env.split(",") if s.strip()]

_freqs_env = os.environ.get("REBAL_FREQS", "1,5,21")
REBAL_FREQS = [int(f.strip()) for f in _freqs_env.split(",") if f.strip()]

REBAL_LOOKBACK = int(os.environ.get("REBAL_LOOKBACK", "20"))

_blends_env = os.environ.get("REBAL_BLENDS", "")
REBAL_BLENDS = [float(a.strip()) for a in _blends_env.split(",") if a.strip()]

TRAINED_STRATEGIES = {"V2_MinVar", "V2_CVaR"}

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
    "evaluate_rebalance.json",
)


def backtest_with_freq(weight_fn, dataset, test_indices, n_assets, freq):
    """Backtest realista con costes y frecuencia de rebalanceo `freq` (en días).

    Entre rebalanceos se mantienen los pesos y derivan con el mercado. El
    coste solo se paga los días de rebalanceo (k % freq == 0, k=0 incluido).
    `w` arranca en Equal-Weight (1/N).

    Args:
        weight_fn: Función `data -> np.ndarray` que devuelve los pesos
            objetivo en un día de rebalanceo.
        dataset: SequenceGraphDataset del que se extraen los grafos de test.
        test_indices: Índices de test sobre `dataset`.
        n_assets: Número de activos de la cartera.
        freq: Frecuencia de rebalanceo en días (1=diario, 5=semanal, 21=mensual).

    Returns:
        tuple[list[float], list[float]]: Retornos netos diarios y turnovers
        diarios (0.0 en los días sin rebalanceo).
    """
    w_held = np.ones(n_assets) / n_assets

    rets_net = []
    turnovers = []

    for k, idx in enumerate(test_indices):
        data = dataset[idx]
        y_simple = np.expm1(data.y.numpy())

        if k % freq == 0:
            w_target = weight_fn(data)
            turnover = float(np.abs(w_target - w_held).sum())
            w_act = w_target
        else:
            turnover = 0.0
            w_act = w_held

        ret_net = float(np.dot(w_act, y_simple) - turnover * TRANSACTION_COST)
        rets_net.append(ret_net)
        turnovers.append(turnover)

        w_drift = w_act * (1.0 + y_simple)
        s = w_drift.sum()
        w_held = w_drift / s if s > 0 else np.ones(n_assets) / n_assets

    return rets_net, turnovers


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


def pack_metrics(rets_net, turnovers):
    """Empaqueta las métricas de una estrategia en el formato de salida del JSON.

    Args:
        rets_net: Secuencia de retornos netos diarios de la estrategia.
        turnovers: Secuencia de turnovers diarios de la estrategia.

    Returns:
        dict: Sharpe, Sortino, Max Drawdown, CVaR-95, retorno anualizado y
        turnover medio diario, todos como floats nativos de Python.
    """
    m = metrics_with_cvar(rets_net)
    return {
        "sharpe_neto": float(m["sharpe_ratio"]),
        "sortino_neto": float(m["sortino_ratio"]),
        "max_drawdown": float(m["max_drawdown"]),
        "cvar95": float(m["cvar95"]),
        "retorno_neto_anualizado": float(m["retorno_anualizado"]),
        "turnover_medio_diario": float(np.mean(turnovers)),
    }


def make_ew_fn(n_assets):
    """Construye la función de pesos Equal-Weight (1/N constante).

    Args:
        n_assets: Número de activos de la cartera.

    Returns:
        Callable[[Any], np.ndarray]: Función `data -> pesos` que ignora
        `data` y siempre devuelve 1/N.
    """
    w = np.ones(n_assets) / n_assets
    return lambda data: w.copy()


def make_hrp_fn():
    """Construye la función de pesos HRP recalculados a partir de `data.cov`.

    Returns:
        Callable[[Any], np.ndarray]: Función `data -> pesos` que aplica
        `hrp_weights` sobre la covarianza del día.
    """
    return lambda data: hrp_weights(data.cov.numpy())


def make_model_fn(model):
    """Construye la función de pesos a partir de un modelo QGNN_V2 ya entrenado.

    Args:
        model: Modelo QGNN_V2 entrenado.

    Returns:
        Callable[[Any], np.ndarray]: Función `data -> pesos` que evalúa el
        modelo en modo inferencia (sin gradiente).
    """
    def _fn(data):
        with torch.no_grad():
            return model(data.x_seq, data.edge_index, data.edge_attr).numpy()
    return _fn


def make_blend_fn(n_assets, alpha):
    """Construye la función de pesos de una mezcla convexa EW + HRP.

    Args:
        n_assets: Número de activos de la cartera.
        alpha: Peso de Equal-Weight en la mezcla; la mezcla es
            `alpha*EW + (1-alpha)*HRP`.

    Returns:
        Callable[[Any], np.ndarray]: Función `data -> pesos` normalizados.
    """
    ew = np.ones(n_assets) / n_assets

    def _fn(data):
        w_hrp = hrp_weights(data.cov.numpy())
        w = alpha * ew + (1.0 - alpha) * w_hrp
        return w / w.sum()
    return _fn


def blend_name(alpha):
    """Genera el nombre de la estrategia de mezcla para un `alpha` dado."""
    return f"Blend_EW{int(round(alpha * 100))}"


def save_incremental(results, setup):
    """Escribe `results` y `setup` en el JSON de salida (_OUTPUT_PATH).

    Args:
        results: Diccionario anidado results[dataset][freq_key][estrategia]
            con los resultados acumulados hasta el momento.
        setup: Diccionario con la configuración del experimento.
    """
    payload = dict(results)
    payload["_setup"] = setup
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=4)


def freq_key(f):
    """Devuelve la clave de diccionario asociada a una frecuencia de rebalanceo."""
    return f"freq_{f}"


def main():
    """Ejecuta el barrido de frecuencia de rebalanceo sobre todas las estrategias y datasets."""
    print("=" * 70, flush=True)
    print("  evaluate_rebalance.py - Barrido de FRECUENCIA de rebalanceo", flush=True)
    print("=" * 70, flush=True)
    print(f"  Epocas: {REBAL_EPOCHS} | Semillas: {REBAL_SEEDS}", flush=True)
    print(f"  Datasets: {REBAL_DATASETS} | Estrategias: {REBAL_STRATEGIES}", flush=True)
    print(f"  Frecuencias (dias): {REBAL_FREQS} | Lookback: {REBAL_LOOKBACK}", flush=True)
    print(f"  TRANSACTION_COST_BPS: {TRANSACTION_COST_BPS}", flush=True)
    print("=" * 70, flush=True)

    setup = {
        "epochs": REBAL_EPOCHS,
        "seeds": REBAL_SEEDS,
        "datasets": REBAL_DATASETS,
        "strategies": REBAL_STRATEGIES,
        "blends_alpha_ew": REBAL_BLENDS,
        "freqs": REBAL_FREQS,
        "lookback": REBAL_LOOKBACK,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "freq_labels": {"1": "diario", "5": "semanal", "21": "mensual"},
    }

    results = {}

    for ds_name in REBAL_DATASETS:
        print(f"\n{'#' * 70}", flush=True)
        print(f"  DATASET: {ds_name}", flush=True)
        print(f"{'#' * 70}", flush=True)

        ds = SequenceGraphDataset(data_dir=ds_name, tau=0.5, lookback=REBAL_LOOKBACK)
        n_assets = ds.num_assets
        train_indices, test_indices, total, train_size, val_size = build_splits(ds)
        print(
            f"  Total grafos: {total} | train_idx: {len(train_indices)} | "
            f"test_idx: {len(test_indices)} | activos: {n_assets}",
            flush=True,
        )

        ds_block = {freq_key(f): {} for f in REBAL_FREQS}
        results[ds_name] = ds_block

        det_fns = {}
        if "Equal_Weight" in REBAL_STRATEGIES:
            det_fns["Equal_Weight"] = make_ew_fn(n_assets)
        if "HRP" in REBAL_STRATEGIES:
            det_fns["HRP"] = make_hrp_fn()
        for alpha in REBAL_BLENDS:
            det_fns[blend_name(alpha)] = make_blend_fn(n_assets, alpha)

        for strat, wfn in det_fns.items():
            print(f"\n--- {strat} (determinista) ---", flush=True)
            for f in REBAL_FREQS:
                t0 = time.time()
                rets, turns = backtest_with_freq(wfn, ds, test_indices, n_assets, f)
                m = pack_metrics(rets, turns)
                ds_block[freq_key(f)][strat] = m
                print(
                    f"  freq={f:>2}d: Sharpe={m['sharpe_neto']:>7.3f} | "
                    f"Sortino={m['sortino_neto']:>7.3f} | "
                    f"MaxDD={m['max_drawdown']:>7.2%} | "
                    f"CVaR95={m['cvar95']:>7.4%} | "
                    f"Turn={m['turnover_medio_diario']:.4f} | "
                    f"{time.time()-t0:.1f}s",
                    flush=True,
                )
                save_incremental(results, setup)

        for strat in REBAL_STRATEGIES:
            if strat not in TRAINED_STRATEGIES:
                continue

            print(f"\n{'=' * 70}", flush=True)
            print(f"  {strat} (entrenada)  dataset={ds_name}", flush=True)
            print("=" * 70, flush=True)

            acc = {
                f: {"sharpe": [], "sortino": [], "maxdd": [], "cvar95": [],
                    "turnover": [], "retan": []}
                for f in REBAL_FREQS
            }

            for seed in REBAL_SEEDS:
                set_global_seed(seed)
                model = QGNN_V2(**QGNN_V2_KWARGS)

                t0 = time.time()
                train_variant(model, ds, train_indices, strat, epochs=REBAL_EPOCHS)
                print(
                    f"  [{strat} seed={seed}] Entrenamiento: "
                    f"{(time.time()-t0)/60:.2f} min",
                    flush=True,
                )

                wfn = make_model_fn(model)
                for f in REBAL_FREQS:
                    rets, turns = backtest_with_freq(wfn, ds, test_indices, n_assets, f)
                    m = metrics_with_cvar(rets)
                    acc[f]["sharpe"].append(float(m["sharpe_ratio"]))
                    acc[f]["sortino"].append(float(m["sortino_ratio"]))
                    acc[f]["maxdd"].append(float(m["max_drawdown"]))
                    acc[f]["cvar95"].append(float(m["cvar95"]))
                    acc[f]["turnover"].append(float(np.mean(turns)))
                    acc[f]["retan"].append(float(m["retorno_anualizado"]))
                    print(
                        f"    freq={f:>2}d seed={seed}: "
                        f"Sharpe={m['sharpe_ratio']:>7.3f} | "
                        f"MaxDD={m['max_drawdown']:>7.2%} | "
                        f"CVaR95={m['cvar95']:>7.4%} | "
                        f"Turn={np.mean(turns):.4f}",
                        flush=True,
                    )

                for f in REBAL_FREQS:
                    a = acc[f]
                    ds_block[freq_key(f)][strat] = {
                        "sharpe_per_seed": list(a["sharpe"]),
                        "sharpe_mean": float(np.mean(a["sharpe"])),
                        "sharpe_std": float(np.std(a["sharpe"])),
                        "sortino_mean": float(np.mean(a["sortino"])),
                        "max_drawdown_mean": float(np.mean(a["maxdd"])),
                        "max_drawdown_std": float(np.std(a["maxdd"])),
                        "cvar95_mean": float(np.mean(a["cvar95"])),
                        "cvar95_std": float(np.std(a["cvar95"])),
                        "turnover_mean": float(np.mean(a["turnover"])),
                        "retorno_neto_anualizado_medio": float(np.mean(a["retan"])),
                        "seeds_hechas": REBAL_SEEDS[: len(a["sharpe"])],
                    }
                save_incremental(results, setup)

            for f in REBAL_FREQS:
                r = ds_block[freq_key(f)][strat]
                print(
                    f"  ==> {strat} freq={f:>2}d: "
                    f"Sharpe={r['sharpe_mean']:.3f}+/-{r['sharpe_std']:.3f} | "
                    f"MaxDD={r['max_drawdown_mean']:.2%} | "
                    f"CVaR95={r['cvar95_mean']:.4%} | "
                    f"Turn={r['turnover_mean']:.4f}",
                    flush=True,
                )

        print("\n" + "=" * 70, flush=True)
        print(f"  TABLA: Sharpe neto por frecuencia — dataset={ds_name}", flush=True)
        print("=" * 70, flush=True)
        strat_order = list(REBAL_STRATEGIES) + [blend_name(a) for a in REBAL_BLENDS]
        cols = "".join([f"{('freq='+str(f)+'d'):>12}" for f in REBAL_FREQS])
        print(f"{'Estrategia':<16}{cols}", flush=True)
        print("-" * (16 + 12 * len(REBAL_FREQS)), flush=True)
        for strat in strat_order:
            row = f"{strat:<16}"
            for f in REBAL_FREQS:
                blk = ds_block[freq_key(f)].get(strat, {})
                val = blk.get("sharpe_mean", blk.get("sharpe_neto"))
                row += f"{(f'{val:.3f}' if val is not None else '--'):>12}"
            print(row, flush=True)
        print("\n  (Turnover medio diario)", flush=True)
        for strat in strat_order:
            row = f"{strat:<16}"
            for f in REBAL_FREQS:
                blk = ds_block[freq_key(f)].get(strat, {})
                val = blk.get("turnover_mean", blk.get("turnover_medio_diario"))
                row += f"{(f'{val:.4f}' if val is not None else '--'):>12}"
            print(row, flush=True)
        print("=" * 70, flush=True)

    print(f"\nResultados guardados en: {_OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
