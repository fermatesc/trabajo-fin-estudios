"""
Harness de backtesting PARAMETRIZABLE y riguroso.

A diferencia de evaluation/backtesting.py (cableado a 4 qubits/L2 y a pesos .pth fijos),
este módulo entrena CUALQUIER configuración desde cero por semilla y la evalúa con el
MISMO protocolo: 5 semillas, pre-entreno en [train_start, test_start), walk-forward con
refit cada 21 días en test, costes de transacción de 10 pb, retornos simples y Rf en las
métricas. Para cada configuración cuántica ejecuta SIEMPRE su gemelo clásico de idéntica
dimensión latente en las mismas condiciones, además del Equal-Weight.

No sobrescribe nada: es una pieza nueva que reutiliza modelos, dataset, loss y métricas
existentes. Guarda los retornos diarios por semilla para los tests de significancia.

Las importaciones de variantes construidas en paralelo (QGNN_V2, QGNN_AmplitudeV3,
SignedFinancialGraphDataset, MultiRelEdgeDataset, etc.) son opcionales y se cargan
de forma perezosa, para que el harness funcione aunque alguna variante no exista
todavía.

Matriz de configuraciones CONFIGS (ver PLAN_EXPERIMENTAL.md); notas sobre las
entradas menos autoexplicativas:
    R8: cierre de la línea temporal, la arquitectura "buena" (GRU + init
        residual 1/N) llevada a 8 qubits, donde la estática (R2) se hundió
        por barren plateaus.
    R9 / R10: sensibilidad al umbral del grafo, tau=0.3 dobla la densidad de
        aristas (28%->52%); R9 estática vs R1, R10 temporal vs R5 (el signo
        es irrelevante, ~0 aristas negativas).
    R11: enriquecimiento de aristas, misma arquitectura temporal "buena"
        (= R5) pero con edge_attr de 3 relaciones [|corr|, signo,
        mismo_sector] en vez de solo |corr|. Aísla el efecto de aristas más
        informativas (beneficia GNN clásica y cuántica por igual: no es
        ventaja cuántica).
    R12: cierre del claim (Opción B), cabeza cuántica inter-activo por
        codificación de amplitud. Front-end temporal idéntico al "bueno"
        (R5: GRU + GAT), pero la cartera entera se codifica como un estado
        conjunto de ceil(log2 N)=5 qubits y se entrelaza antes de medir
        (regla de Born), contrastando la hipótesis de entrelazamiento entre
        activos (no por activo). El gemelo clásico (ClassicalAmplitudeV3)
        mezcla los mismos N scores de forma conjunta (capa densa + softmax),
        aislando "mezcla cuántica vs mezcla clásica".

Uso:
    python evaluation/backtest_harness.py --config R1
    python evaluation/backtest_harness.py --config R1 --smoke      # rápido (1 semilla, 2 épocas)
    python evaluation/backtest_harness.py --config all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (set_global_seed, IN_CHANNELS, HIDDEN_CHANNELS,
                    LEARNING_RATE, TRANSACTION_COST_BPS, TRAIN_RATIO, VAL_RATIO)
from data_utils.dataset_loader import FinancialGraphDataset
from data_utils.sequence_dataset import SequenceGraphDataset
from models.qgnn_model import QGNN_Portfolio
from models.classical_gnn_model import ClassicalGNN_Portfolio
from models.loss_functions import DifferentiableSharpeLoss
from evaluation.metrics import calculate_portfolio_metrics

try:
    from models.qgnn_v2 import QGNN_V2
except Exception:
    QGNN_V2 = None
try:
    from models.classical_v2 import ClassicalGNN_V2
except Exception:
    ClassicalGNN_V2 = None
try:
    from models.qgnn_amplitude import QGNN_AmplitudeV3
except Exception:
    QGNN_AmplitudeV3 = None
try:
    from models.classical_amplitude import ClassicalAmplitudeV3
except Exception:
    ClassicalAmplitudeV3 = None
try:
    from data_utils.signed_dataset import SignedFinancialGraphDataset
except Exception:
    SignedFinancialGraphDataset = None
try:
    from data_utils.multirel_dataset import MultiRelEdgeDataset, EDGE_DIM as MULTIREL_EDGE_DIM
except Exception:
    MultiRelEdgeDataset = None
    MULTIREL_EDGE_DIM = 3

SEEDS = [42, 43, 44, 45, 46]
REFIT_FREQ = 21
EPOCHS_REFIT = 3
PRETRAIN_EPOCHS = 15

CONFIGS = {
    "R0": dict(kind="qgnn",    n_qubits=4, q_layers=2,  temp=5.0),
    "R1": dict(kind="qgnn",    n_qubits=6, q_layers=2,  temp=5.0),
    "R2": dict(kind="qgnn",    n_qubits=8, q_layers=2,  temp=5.0),
    "R3a": dict(kind="qgnn",   n_qubits=4, q_layers=4,  temp=5.0),
    "R3b": dict(kind="qgnn",   n_qubits=4, q_layers=6,  temp=5.0),
    "R4": dict(kind="qgnn_v2", n_qubits=4, q_layers=2,  temp=1.5, reupload=True, gru_hidden=16, lookback=20),
    "R5": dict(kind="qgnn_v2", n_qubits=6, q_layers=2,  temp=1.5, reupload=True, gru_hidden=16, lookback=20),
    "R6": dict(kind="qgnn",    n_qubits=6, q_layers=2,  temp=5.0, signed=True),
    "R7": dict(kind="qgnn",    n_qubits=6, q_layers=2,  temp=1.5),
    "R8": dict(kind="qgnn_v2", n_qubits=8, q_layers=2,  temp=1.5, reupload=True, gru_hidden=16, lookback=20),
    "R9":  dict(kind="qgnn",    n_qubits=6, q_layers=2,  temp=5.0, tau=0.3),
    "R10": dict(kind="qgnn_v2", n_qubits=6, q_layers=2,  temp=1.5, reupload=True, gru_hidden=16, lookback=20, tau=0.3),
    "R11": dict(kind="qgnn_v2", n_qubits=6, q_layers=2,  temp=1.5, reupload=True, gru_hidden=16, lookback=20, edge="multirel"),
    "R12": dict(kind="qgnn_amp", latent_dim=4, q_layers=2, temp=1.5,
                gru_hidden=16, lookback=20),
}


def _defaults(cfg):
    """Completa una configuración de CONFIGS con sus valores por defecto.

    Args:
        cfg: dict parcial de configuración (una entrada de CONFIGS).

    Returns:
        dict con todos los campos rellenados, más los derivados `edge_dim`
        (dimensión de edge_attr: 1 para "base" [|corr|], 3 para "multirel"
        [|corr|, signo, mismo_sector]) y `temporal` (True si el modelo
        consume secuencias x_seq, es decir kind en qgnn_v2/qgnn_amp).

    Notes:
        `latent_dim` y `mix_hidden` solo aplican a kind="qgnn_amp": dimensión
        latente del GAT antes del score cuántico y cuello de botella del
        gemelo clásico (None = capa densa plena), respectivamente.
        `n_assets` se deja en None aquí y se inyecta en run_config una vez
        construido el dataset (la cabeza de amplitud fija el nº de qubits
        a partir del nº de activos).
    """
    c = dict(cfg)
    c.setdefault("hidden", HIDDEN_CHANNELS)
    c.setdefault("q_layers", 2)
    c.setdefault("reupload", True)
    c.setdefault("gru_hidden", 16)
    c.setdefault("lookback", 20)
    c.setdefault("signed", False)
    c.setdefault("tau", 0.5)
    c.setdefault("edge", "base")
    c.setdefault("latent_dim", 4)
    c.setdefault("mix_hidden", None)
    c.setdefault("n_assets", None)
    c["edge_dim"] = MULTIREL_EDGE_DIM if c["edge"] == "multirel" else 1
    c["temporal"] = c["kind"] in ("qgnn_v2", "qgnn_amp")
    return c


def build_dataset(cfg, data_dir="data"):
    """Construye el dataset adecuado a la configuración dada.

    Comparte la misma base para todas las variantes (temporal o no, signed o
    con edges multirelacionales).

    Args:
        cfg: dict de configuración ya expandido por `_defaults`.
        data_dir: directorio del dataset; permite apuntar a uno alternativo
            (p. ej. data_xl/ con la historia extendida 2005-2024) sin tocar
            el dataset por defecto.

    Returns:
        Tupla (dataset, temporal_flag).
    """
    tau = cfg["tau"]
    edge = cfg.get("edge", "base")

    def _base(t):
        if cfg["signed"]:
            if SignedFinancialGraphDataset is None:
                raise RuntimeError("signed=True pero SignedFinancialGraphDataset no disponible")
            return SignedFinancialGraphDataset(data_dir=data_dir, tau=t)
        if edge == "multirel":
            if MultiRelEdgeDataset is None:
                raise RuntimeError("edge=multirel pero MultiRelEdgeDataset no disponible")
            return MultiRelEdgeDataset(data_dir=data_dir, tau=t)
        return FinancialGraphDataset(data_dir=data_dir, tau=t)

    if cfg["temporal"]:
        ds = SequenceGraphDataset(data_dir=data_dir, tau=tau, lookback=cfg["lookback"])
        if cfg["signed"] or edge == "multirel":
            ds.base = _base(tau)
        return ds, True
    return _base(tau), False


def build_quantum(cfg):
    """Instancia el modelo cuántico correspondiente a cfg["kind"].

    Args:
        cfg: dict de configuración ya expandido por `_defaults`.

    Returns:
        Instancia de QGNN_Portfolio, QGNN_V2 o QGNN_AmplitudeV3 según el caso.

    Raises:
        RuntimeError: si la variante requerida no está disponible o falta
            `n_assets` para qgnn_amp.
        ValueError: si cfg["kind"] no es reconocido.
    """
    if cfg["kind"] == "qgnn":
        return QGNN_Portfolio(in_channels=IN_CHANNELS, hidden_channels=cfg["hidden"],
                              n_qubits=cfg["n_qubits"], q_layers=cfg["q_layers"],
                              temperature=cfg["temp"])
    elif cfg["kind"] == "qgnn_v2":
        if QGNN_V2 is None:
            raise RuntimeError("QGNN_V2 no disponible")
        return QGNN_V2(in_channels=IN_CHANNELS, hidden_channels=cfg["hidden"],
                       n_qubits=cfg["n_qubits"], q_layers=cfg["q_layers"],
                       gru_hidden=cfg["gru_hidden"], temperature=cfg["temp"],
                       reupload=cfg["reupload"], edge_dim=cfg["edge_dim"])
    elif cfg["kind"] == "qgnn_amp":
        if QGNN_AmplitudeV3 is None:
            raise RuntimeError("QGNN_AmplitudeV3 no disponible")
        if cfg["n_assets"] is None:
            raise RuntimeError("qgnn_amp requiere n_assets (se inyecta en run_config)")
        return QGNN_AmplitudeV3(n_assets=cfg["n_assets"], in_channels=IN_CHANNELS,
                                hidden_channels=cfg["hidden"], latent_dim=cfg["latent_dim"],
                                q_layers=cfg["q_layers"], gru_hidden=cfg["gru_hidden"],
                                edge_dim=cfg["edge_dim"])
    raise ValueError(cfg["kind"])


def build_classical(cfg):
    """Gemelo clásico de idéntica dimensión latente (= n_qubits) y misma temperatura."""
    if cfg["kind"] == "qgnn":
        return ClassicalGNN_Portfolio(in_channels=IN_CHANNELS, hidden_channels=cfg["hidden"],
                                      latent_dim=cfg["n_qubits"], temperature=cfg["temp"])
    elif cfg["kind"] == "qgnn_v2":
        if ClassicalGNN_V2 is None:
            raise RuntimeError("ClassicalGNN_V2 no disponible")
        return ClassicalGNN_V2(in_channels=IN_CHANNELS, hidden_channels=cfg["hidden"],
                               latent_dim=cfg["n_qubits"], gru_hidden=cfg["gru_hidden"],
                               temperature=cfg["temp"], edge_dim=cfg["edge_dim"])
    elif cfg["kind"] == "qgnn_amp":
        if ClassicalAmplitudeV3 is None:
            raise RuntimeError("ClassicalAmplitudeV3 no disponible")
        if cfg["n_assets"] is None:
            raise RuntimeError("qgnn_amp requiere n_assets (se inyecta en run_config)")
        return ClassicalAmplitudeV3(n_assets=cfg["n_assets"], in_channels=IN_CHANNELS,
                                    hidden_channels=cfg["hidden"], latent_dim=cfg["latent_dim"],
                                    gru_hidden=cfg["gru_hidden"], temperature=cfg["temp"],
                                    mix_hidden=cfg["mix_hidden"], edge_dim=cfg["edge_dim"])
    raise ValueError(cfg["kind"])


def fwd(model, sample, temporal):
    """Forward pass del modelo, usando x_seq si es temporal o x en caso contrario."""
    if temporal:
        return model(sample.x_seq, sample.edge_index, sample.edge_attr)
    return model(sample.x, sample.edge_index, sample.edge_attr)


def train_range(model, dataset, lo, hi, epochs, temporal, risk_mode="std"):
    """Entrena `model` sobre las muestras [lo, hi) del dataset.

    Args:
        model: modelo a entrenar (cuántico o clásico).
        dataset: dataset indexable con muestras que exponen x/x_seq,
            edge_index, edge_attr, y y cov.
        lo: índice inicial (inclusive) del rango de entrenamiento.
        hi: índice final (exclusive) del rango de entrenamiento.
        epochs: número de épocas sobre el rango.
        temporal: True si el modelo consume secuencias (x_seq).
        risk_mode: modo de riesgo pasado a `DifferentiableSharpeLoss`.

    Returns:
        El modelo entrenado (mismo objeto, mutado in-place).
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DifferentiableSharpeLoss(risk_free_rate=0.0, risk_mode=risk_mode)
    model.train()
    for _ in range(epochs):
        for i in range(lo, hi):
            sample = dataset[i]
            optimizer.zero_grad()
            weights = fwd(model, sample, temporal)
            loss = criterion(weights, sample.y, sample.cov)
            loss.backward()
            optimizer.step()
    return model


def walk_forward(model, dataset, test_indices, train_start, test_start,
                 refit_epochs, refit_freq, temporal, risk_mode="std"):
    """Evalúa `model` en walk-forward sobre `test_indices`, con refit periódico.

    Cada `refit_freq` días re-entrena el modelo sobre el tramo transcurrido
    desde el último refit, descuenta costes de transacción proporcionales al
    turnover y actualiza los pesos por deriva (drift) con los retornos
    realizados antes del siguiente rebalanceo.

    Args:
        model: modelo a evaluar (cuántico o clásico), ya pre-entrenado.
        dataset: dataset indexable con muestras que exponen x/x_seq,
            edge_index, edge_attr e y.
        test_indices: lista de índices del dataset a recorrer en orden.
        train_start: índice inicial usado en el pre-entreno (no usado
            directamente aquí, se mantiene por compatibilidad de firma).
        test_start: índice donde comienza el tramo de test.
        refit_epochs: épocas de entrenamiento en cada refit.
        refit_freq: cada cuántos días de test se hace un refit.
        temporal: True si el modelo consume secuencias (x_seq).
        risk_mode: modo de riesgo pasado a `DifferentiableSharpeLoss` en el refit.

    Returns:
        Tupla (daily_simple_returns_net, mean_turnover): array de retornos
        simples diarios netos de costes, y turnover medio a lo largo del
        tramo de test.
    """
    N = dataset.num_assets
    w_prev = (np.ones(N) / N)
    rets, turns = [], []
    current_train_end = test_start
    for idx, i in enumerate(test_indices):
        if idx > 0 and idx % refit_freq == 0:
            model = train_range(model, dataset, current_train_end, i, refit_epochs, temporal, risk_mode)
            current_train_end = i
        model.eval()
        sample = dataset[i]
        with torch.no_grad():
            w = fwd(model, sample, temporal).numpy()
        y_simple = np.expm1(sample.y.numpy())
        turnover = np.abs(w - w_prev).sum()
        ret = np.dot(w, y_simple) - (turnover * TRANSACTION_COST_BPS / 10000.0)
        rets.append(float(ret))
        turns.append(float(turnover))
        w_drift = w * (1.0 + y_simple)
        w_prev = w_drift / w_drift.sum()
    return np.array(rets), float(np.mean(turns))


def equal_weight_returns(dataset, test_indices):
    """Calcula los retornos diarios netos de costes de la cartera Equal-Weight.

    Args:
        dataset: dataset indexable con muestras que exponen y (retornos log).
        test_indices: lista de índices del dataset a recorrer en orden.

    Returns:
        Tupla (daily_simple_returns_net, mean_turnover), igual que `walk_forward`.
    """
    N = dataset.num_assets
    w_target = np.ones(N) / N
    w_prev = w_target.copy()
    rets, turns = [], []
    for i in test_indices:
        sample = dataset[i]
        y_simple = np.expm1(sample.y.numpy())
        turnover = np.abs(w_target - w_prev).sum()
        ret = np.dot(w_target, y_simple) - (turnover * TRANSACTION_COST_BPS / 10000.0)
        rets.append(float(ret))
        turns.append(float(turnover))
        w_drift = w_target * (1.0 + y_simple)
        w_prev = w_drift / w_drift.sum()
    return np.array(rets), float(np.mean(turns))


def run_config(config_id, cfg, seeds, pretrain_epochs, refit_epochs, refit_freq, risk_mode="std",
               data_dir="data", train_ratio=None):
    """Ejecuta el protocolo completo (pre-entreno + walk-forward) para una configuración.

    Entrena y evalúa el modelo cuántico y su gemelo clásico para cada semilla
    de `seeds`, además de la cartera Equal-Weight, y agrega los resultados en
    un ensemble por semillas. Usa un checkpoint en disco (`ckpt_<config_id>_<risk_mode>[_<data_tag>].json`)
    para poder reanudar tras cortes o suspensión sin repetir semillas ya completadas.

    Args:
        config_id: identificador de la configuración (clave de CONFIGS).
        cfg: dict de configuración (una entrada de CONFIGS, sin expandir).
        seeds: lista de semillas a ejecutar.
        pretrain_epochs: épocas de pre-entreno antes del walk-forward.
        refit_epochs: épocas de cada refit durante el walk-forward.
        refit_freq: cada cuántos días de test se hace un refit.
        risk_mode: modo de riesgo de la loss ("std" o "cov").
        data_dir: directorio del dataset (p. ej. "data" o "data_xl").
        train_ratio: override de TRAIN_RATIO; si es None se usa TRAIN_RATIO global.

    Returns:
        dict con la configuración expandida, el nº de días de test, y las
        métricas/retornos diarios (por semilla y ensemble) de quantum,
        classical y equal_weight.
    """
    cfg = _defaults(cfg)
    dataset, temporal = build_dataset(cfg, data_dir=data_dir)
    cfg["n_assets"] = dataset.num_assets
    tr = TRAIN_RATIO if train_ratio is None else train_ratio
    print(f"\n{'='*70}\nCONFIG {config_id} [loss={risk_mode}] [data={data_dir} train_ratio={tr}]: {cfg}\n{'='*70}")
    total = len(dataset)
    train_start = (cfg["lookback"] - 1) if temporal else 0
    test_start = int(total * tr)
    test_indices = list(range(test_start, total))
    print(f"total={total} train_start={train_start} test_start={test_start} "
          f"test_days={len(test_indices)} temporal={temporal}")

    data_tag = os.path.basename(os.path.normpath(data_dir))
    data_suffix = "" if data_tag == "data" else f"_{data_tag}"
    ckpt_path = f"ckpt_{config_id}_{risk_mode}{data_suffix}.json"
    ck = {"seeds_done": [], "q_per_seed": [], "c_per_seed": [],
          "q_turns": [], "c_turns": [], "ew_rets": None, "ew_turn": None}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                ck = json.load(f)
            print(f"Checkpoint encontrado: semillas ya hechas {ck['seeds_done']}")
        except Exception:
            pass

    if ck["ew_rets"] is None:
        ew_rets, ew_turn = equal_weight_returns(dataset, test_indices)
        ck["ew_rets"] = ew_rets.tolist(); ck["ew_turn"] = ew_turn
    else:
        ew_rets = np.array(ck["ew_rets"]); ew_turn = ck["ew_turn"]

    for seed in seeds:
        if seed in ck["seeds_done"]:
            print(f"\n--- semilla {seed} ya en checkpoint, se omite ---")
            continue
        t0 = time.time()
        print(f"\n--- semilla {seed} ---")
        set_global_seed(seed)
        qm = build_quantum(cfg)
        set_global_seed(seed)
        cm = build_classical(cfg)

        qm = train_range(qm, dataset, train_start, test_start, pretrain_epochs, temporal, risk_mode)
        cm = train_range(cm, dataset, train_start, test_start, pretrain_epochs, temporal, risk_mode)

        qr, qt = walk_forward(qm, dataset, test_indices, train_start, test_start,
                              refit_epochs, refit_freq, temporal, risk_mode)
        cr, ct = walk_forward(cm, dataset, test_indices, train_start, test_start,
                              refit_epochs, refit_freq, temporal, risk_mode)
        ck["q_per_seed"].append(qr.tolist()); ck["c_per_seed"].append(cr.tolist())
        ck["q_turns"].append(qt); ck["c_turns"].append(ct)
        ck["seeds_done"].append(seed)
        with open(ckpt_path, "w") as f:
            json.dump(ck, f)
        qs = calculate_portfolio_metrics(qr, benchmark_returns=ew_rets)["sharpe_ratio"]
        cs = calculate_portfolio_metrics(cr, benchmark_returns=ew_rets)["sharpe_ratio"]
        print(f"  Sharpe Q={qs:.3f}  C={cs:.3f}  ({time.time()-t0:.0f}s)  [checkpoint guardado]")

    q_per_seed = ck["q_per_seed"]; c_per_seed = ck["c_per_seed"]
    q_turns = ck["q_turns"]; c_turns = ck["c_turns"]

    def summarize(per_seed, turns):
        """Agrega los retornos por semilla en un ensemble y calcula métricas."""
        arr = np.array(per_seed)
        ens = arr.mean(axis=0)
        m = calculate_portfolio_metrics(ens, benchmark_returns=ew_rets)
        m["turnover_medio"] = float(np.mean(turns))
        per = [calculate_portfolio_metrics(np.array(r), benchmark_returns=ew_rets)["sharpe_ratio"]
               for r in per_seed]
        m["sharpe_per_seed"] = per
        m["sharpe_seed_mean"] = float(np.mean(per))
        m["sharpe_seed_std"] = float(np.std(per))
        return m, ens.tolist()

    q_metrics, q_ens = summarize(q_per_seed, q_turns)
    c_metrics, c_ens = summarize(c_per_seed, c_turns)
    ew_metrics = calculate_portfolio_metrics(ew_rets, benchmark_returns=ew_rets)
    ew_metrics["turnover_medio"] = ew_turn

    print(f"\n[{config_id}] Sharpe ensemble  Q={q_metrics['sharpe_ratio']:.3f}  "
          f"C={c_metrics['sharpe_ratio']:.3f}  EW={ew_metrics['sharpe_ratio']:.3f}")
    print(f"[{config_id}] Sharpe seed mean+/-std  "
          f"Q={q_metrics['sharpe_seed_mean']:.3f}+/-{q_metrics['sharpe_seed_std']:.3f}  "
          f"C={c_metrics['sharpe_seed_mean']:.3f}+/-{c_metrics['sharpe_seed_std']:.3f}")

    return {
        "config": cfg,
        "test_days": len(test_indices),
        "quantum": {"metrics": q_metrics, "per_seed_daily_returns": q_per_seed,
                    "ensemble_daily_returns": q_ens},
        "classical": {"metrics": c_metrics, "per_seed_daily_returns": c_per_seed,
                      "ensemble_daily_returns": c_ens},
        "equal_weight": {"metrics": ew_metrics, "daily_returns": ew_rets.tolist()},
    }


def main():
    """CLI del harness: parsea argumentos, ejecuta las configs pedidas y
    acumula resultados en un único JSON de salida (--out)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="R1", help="ID de CONFIGS o 'all'")
    ap.add_argument("--smoke", action="store_true", help="1 semilla, 2 épocas pre-entreno")
    ap.add_argument("--nseeds", type=int, default=0, help="override nº de semillas (0=todas)")
    ap.add_argument("--pretrain-epochs", type=int, default=0, help="override épocas pre-entreno (0=default)")
    ap.add_argument("--loss", choices=["std", "cov"], default="std",
                    help="modo de riesgo en la loss: std (auditoría) | cov (sqrt(w'Sigma w) con shrinkage)")
    ap.add_argument("--data-dir", default="data",
                    help="directorio del dataset (p. ej. data_xl para la historia extendida 2005-2024)")
    ap.add_argument("--train-ratio", type=float, default=None,
                    help="override de TRAIN_RATIO; bajarlo (p. ej. 0.25) adelanta el inicio del "
                         "tramo OOS para que el walk-forward cubra 2008 y 2020")
    ap.add_argument("--out", default="results_harness.json")
    args = ap.parse_args()

    if args.smoke:
        seeds, pre, refit, freq = [42], 2, 1, 21
    else:
        seeds, pre, refit, freq = SEEDS, PRETRAIN_EPOCHS, EPOCHS_REFIT, REFIT_FREQ
    if args.nseeds:
        seeds = SEEDS[:args.nseeds]
    if args.pretrain_epochs:
        pre = args.pretrain_epochs

    ids = list(CONFIGS.keys()) if args.config == "all" else [args.config]

    out = {}
    if os.path.exists(args.out):
        try:
            with open(args.out) as f:
                out = json.load(f)
        except Exception:
            out = {}

    for cid in ids:
        if cid not in CONFIGS:
            print(f"Config desconocida: {cid}"); continue
        res = run_config(cid, CONFIGS[cid], seeds, pre, refit, freq, args.loss,
                         data_dir=args.data_dir, train_ratio=args.train_ratio)
        key = cid + ("_smoke" if args.smoke else "")
        out[key] = res
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Guardado {key} en {args.out}")


if __name__ == "__main__":
    main()
