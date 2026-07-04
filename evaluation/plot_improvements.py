"""
Genera la figura comparativa del programa de mejora predictiva (variantes QGNN-v2)
para el Capítulo 5 del TFM. Lee evaluate_all.json y produce un panel doble:
  (a) data_large (62 activos, periodo alcista)
  (b) data_xl    (58 activos, 2005-2024, con crisis)
Cada panel: Sharpe neto de las 3 variantes frente a la línea de Equal-Weight.
Los colores de las barras codifican el tipo de variante: rojo = orientada a
retorno, gris = ancla EW, verde = orientada a riesgo.
Salida: figuras/ del proyecto LaTeX.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, "..", "..", "Plantilla LATEX TFE TFM UNIR", "figuras")

with open(os.path.join(REPO, "evaluate_all.json")) as f:
    d = json.load(f)

variants = ["V2_WinSharpe", "V2_WinSharpeAnc", "V2_MinVar"]
vlabels = ["WinSharpe\n(retorno)", "WinSharpeAnc\n(ancla EW)", "MinVar\n(riesgo)"]
vcolors = ["#C0392B", "#7F8C8D", "#1E8449"]

datasets = [
    ("data_large", "(a) 62 activos (2019-2024, alcista)"),
    ("data_xl", "(b) 58 activos (2005-2024, con crisis)"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

for ax, (dsname, title) in zip(axes, datasets):
    block = d[dsname]
    means = [block[v]["sharpe_mean"] for v in variants]
    errs = [block[v]["sharpe_std"] for v in variants]
    ew = block["Equal_Weight"]["sharpe_neto"]

    x = np.arange(len(variants))
    ax.bar(x, means, 0.6, yerr=errs, capsize=4, color=vcolors, edgecolor="black", linewidth=0.4)
    ax.axhline(ew, color="#D35400", linestyle="--", linewidth=2, label=f"Equal-Weight ({ew:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(vlabels, fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", fontsize=8.5)
    top = max(max(means), ew) * 1.18
    ax.set_ylim(0, top)

axes[0].set_ylabel("Ratio de Sharpe neto")
plt.tight_layout()
out = os.path.normpath(os.path.join(FIG_DIR, "figura_mejora_variantes.png"))
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figura guardada en: {out}")
