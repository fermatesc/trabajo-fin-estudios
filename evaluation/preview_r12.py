"""Adelanto rápido del veredicto de R12 (Opción B). Lee el JSON de resultados."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "results_harness_cov.json"
d = json.load(open(path))
if "R12" not in d:
    print(f"R12 aún no está en {path}")
    sys.exit(0)
r = d["R12"]
m, c, e = r["quantum"]["metrics"], r["classical"]["metrics"], r["equal_weight"]["metrics"]
print(f"=== R12 ({path}) — capa cuantica INTER-ACTIVO (amplitude encoding, 5 qubits) ===")
print(f"  Q  ensemble Sharpe = {m['sharpe_ratio']:.3f}   por-semilla {m['sharpe_seed_mean']:.3f} +/- {m['sharpe_seed_std']:.3f}")
print(f"  C  ensemble Sharpe = {c['sharpe_ratio']:.3f}   por-semilla {c['sharpe_seed_mean']:.3f} +/- {c['sharpe_seed_std']:.3f}")
print(f"  EW ensemble Sharpe = {e['sharpe_ratio']:.3f}")
print(f"  turnover  Q={m.get('turnover_medio',0):.3f}  C={c.get('turnover_medio',0):.3f}  EW={e.get('turnover_medio',0):.3f}")
print(f"  Sharpe por semilla Q: {[round(x,3) for x in m.get('sharpe_per_seed',[])]}")
print(f"  Sharpe por semilla C: {[round(x,3) for x in c.get('sharpe_per_seed',[])]}")
