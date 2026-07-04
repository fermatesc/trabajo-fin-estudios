"""
Lanzador paralelo de la batería de backtesting.

Ejecuta todas las combinaciones (config × loss) como subprocesos independientes
del harness (evaluation/backtest_harness.py), con un tope de concurrencia y pocos
hilos por proceso para repartir los núcleos sin sobre-suscribir. Cada subproceso
ya hace checkpoint por config+loss (ckpt_<cfg>_<loss>.json), de modo que el
conjunto es reanudable: relanzar salta las semillas ya hechas.

Al terminar, fusiona los ficheros parciales en results_harness_cov.json y
results_harness_std.json (claves R0..R11), listos para evaluation/analyze_harness.py.

Uso:
    python evaluation/run_battery.py                       # todas las configs, cov+std
    python evaluation/run_battery.py --concurrency 7 --threads 2
    python evaluation/run_battery.py --configs R4,R5 --losses cov
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTS_DIR = os.path.join(ROOT, "results_parts")

ALL_CONFIGS = ["R0", "R1", "R2", "R3a", "R3b", "R4", "R5", "R6", "R7", "R8",
               "R9", "R10", "R11"]


def part_path(cfg, loss):
    return os.path.join(PARTS_DIR, f"{cfg}_{loss}.json")


def is_done(cfg, loss, nseeds):
    """Completo si el fichero parcial existe y la config tiene todas las semillas."""
    p = part_path(cfg, loss)
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
        res = d.get(cfg)
        if not res:
            return False
        n = len(res["quantum"]["per_seed_daily_returns"])
        return n >= nseeds
    except Exception:
        return False


def launch(cfg, loss, threads):
    """Lanza el harness como subproceso para (cfg, loss), con log propio en results_parts/."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = str(threads)
    log = open(os.path.join(PARTS_DIR, f"{cfg}_{loss}.log"), "w")
    cmd = [sys.executable, "-u", os.path.join("evaluation", "backtest_harness.py"),
           "--config", cfg, "--loss", loss, "--out", part_path(cfg, loss)]
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    p._logfile = log
    return p


def merge(losses):
    """Fusiona los JSON parciales de results_parts/ en results_harness_<loss>.json."""
    for loss in losses:
        out = {}
        for cfg in ALL_CONFIGS:
            p = part_path(cfg, loss)
            if os.path.exists(p):
                try:
                    out.update(json.load(open(p)))
                except Exception as e:
                    print(f"  aviso: no se pudo leer {p}: {e}")
        dest = os.path.join(ROOT, f"results_harness_{loss}.json")
        json.dump(out, open(dest, "w"), indent=2)
        print(f"  fusionado {len(out)} configs -> {dest}")


def main():
    """CLI: lanza en paralelo todos los jobs (config, loss) pendientes,
    respetando el tope de concurrencia, y fusiona los resultados al terminar.

    Notes:
        Los jobs se ordenan para lanzar primero las configs temporales/grandes
        (más lentas: R4, R5, R8, R10, R11) y lograr un mejor empaquetado de
        la cola de concurrencia.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=7)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS))
    ap.add_argument("--losses", default="cov,std")
    args = ap.parse_args()

    os.makedirs(PARTS_DIR, exist_ok=True)
    configs = args.configs.split(",")
    losses = args.losses.split(",")
    jobs = [(c, l) for l in losses for c in configs]
    slow = {"R5", "R4", "R8", "R10", "R11"}
    jobs.sort(key=lambda cl: (cl[0] not in slow))

    pending = [j for j in jobs if not is_done(*j, args.nseeds)]
    done0 = len(jobs) - len(pending)
    print(f"Batería: {len(jobs)} jobs ({len(losses)} losses × {len(configs)} configs), "
          f"{done0} ya completos, {len(pending)} por correr. "
          f"concurrency={args.concurrency} threads={args.threads}")

    running = {}
    t0 = time.time()
    idx = 0
    completed = done0
    total = len(jobs)
    while idx < len(pending) or running:
        while idx < len(pending) and len(running) < args.concurrency:
            cfg, loss = pending[idx]; idx += 1
            proc = launch(cfg, loss, args.threads)
            running[proc] = (cfg, loss)
            print(f"[{time.strftime('%H:%M:%S')}] lanzado {cfg}/{loss} "
                  f"(corriendo {len(running)})")
        time.sleep(5)
        for proc in list(running):
            if proc.poll() is not None:
                cfg, loss = running.pop(proc)
                proc._logfile.close()
                completed += 1
                ok = is_done(cfg, loss, args.nseeds)
                mins = (time.time() - t0) / 60
                print(f"[{time.strftime('%H:%M:%S')}] TERMINADO {cfg}/{loss} "
                      f"rc={proc.returncode} done={'OK' if ok else 'FALLO'} "
                      f"({completed}/{total}, {mins:.0f} min)")

    print("\nFusionando ficheros parciales...")
    merge(losses)
    print(f"Batería completa en {(time.time()-t0)/60:.0f} min.")


if __name__ == "__main__":
    main()
