"""Mide el sobrecoste del forward: QGNN_AmplitudeV3 vs ClassicalAmplitudeV3."""
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.sequence_dataset import SequenceGraphDataset
from models.qgnn_amplitude import QGNN_AmplitudeV3
from models.classical_amplitude import ClassicalAmplitudeV3

torch.manual_seed(42)
ds = SequenceGraphDataset(data_dir="data", tau=0.5, lookback=20)
s = ds.get(50)
qm = QGNN_AmplitudeV3(n_assets=ds.num_assets).eval()
cm = ClassicalAmplitudeV3(n_assets=ds.num_assets).eval()

def bench(model, reps=200):
    """Mide el tiempo medio de forward de un modelo sobre una muestra fija.

    Args:
        model: Modelo a evaluar (con método ``forward`` compatible con
            ``s.x_seq, s.edge_index, s.edge_attr``).
        reps: Número de repeticiones sobre las que promediar el tiempo.

    Returns:
        Tiempo medio de forward en milisegundos.
    """
    with torch.no_grad():
        for _ in range(20):
            model(s.x_seq, s.edge_index, s.edge_attr)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(s.x_seq, s.edge_index, s.edge_attr)
        return (time.perf_counter() - t0) / reps * 1000.0

tq = bench(qm)
tc = bench(cm)
nq = sum(p.numel() for p in qm.parameters())
ncl = sum(p.numel() for p in cm.parameters())
print(f"Forward QGNN_AmplitudeV3 (cuantico) : {tq:.2f} ms  | params={nq} (cuanticos={qm.q_weights.numel()})")
print(f"Forward ClassicalAmplitudeV3 (gemelo): {tc:.2f} ms  | params={ncl}")
print(f"Factor de sobrecoste cuantico/clasico = x{tq/tc:.1f}")
