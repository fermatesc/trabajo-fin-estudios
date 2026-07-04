"""
Genera la figura de la línea de RIESGO para el Capítulo 5 del TFM.
Lee evaluate_rebalance.json y produce un panel doble:
  (a) Frontera riesgo/retorno del blend EW⊕HRP (rebalanceo mensual, data_xl):
      Sharpe neto frente a CVaR-95, interpolando de Equal-Weight a HRP. El eje
      X se invierte para que un menor CVaR (mejor cola) quede a la derecha.
  (b) Efecto de la frecuencia de rebalanceo sobre HRP (Sharpe vs diario/
      semanal/mensual) en ambos datasets: muestra cómo la frecuencia
      convierte una estrategia inviable a diario en competitiva.
Salida: figuras/ del proyecto LaTeX (figura_riesgo_rebalanceo.png).

En el panel (a), las etiquetas de cada punto llevan desplazamientos
individuales para evitar solapes, y se resaltan con un círculo Equal-Weight
(100% EW) y el blend recomendado (Blend_EW75).
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, "..", "..", "Plantilla LATEX TFE TFM UNIR", "figuras")

with open(os.path.join(REPO, "evaluate_rebalance.json")) as f:
    d = json.load(f)


def sharpe(ds, freq, strat):
    """Sharpe neto de `strat` en el dataset/frecuencia dados (media si hay varias semillas)."""
    blk = d[ds][f"freq_{freq}"][strat]
    return blk.get("sharpe_mean", blk.get("sharpe_neto"))


def cvar(ds, freq, strat):
    """CVaR-95 de `strat` en el dataset/frecuencia dados (media si hay varias semillas)."""
    blk = d[ds][f"freq_{freq}"][strat]
    return blk.get("cvar95_mean", blk.get("cvar95"))


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

ax = axes[0]
order = ["Equal_Weight", "Blend_EW75", "Blend_EW50", "Blend_EW25", "HRP"]
labels = ["EW\n(100%)", "Blend 75", "Blend 50", "Blend 25", "HRP\n(0%)"]
xs = [cvar("data_xl", 21, s) * 100 for s in order]
ys = [sharpe("data_xl", 21, s) for s in order]

ax.plot(xs, ys, "-o", color="#1F618D", linewidth=1.8, markersize=7,
        markeredgecolor="black", markeredgewidth=0.5, zorder=3)
offs = [(10, -16), (7, 6), (7, 6), (7, 6), (-4, 10)]
has = ["left", "left", "left", "left", "right"]
for x, y, lab, off, ha in zip(xs, ys, labels, offs, has):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off,
                fontsize=8.2, ha=ha)
ax.margins(y=0.16)
ax.scatter([xs[0]], [ys[0]], s=120, facecolors="none",
           edgecolors="#D35400", linewidths=2, zorder=4)
ax.scatter([xs[1]], [ys[1]], s=120, facecolors="none",
           edgecolors="#1E8449", linewidths=2, zorder=4)
ax.set_xlabel("CVaR-95 diario (%)  →  más riesgo de cola")
ax.set_ylabel("Ratio de Sharpe neto")
ax.set_title("(a) Frontera EW⊕HRP — historia extendida (mensual)", fontsize=10.5)
ax.grid(alpha=0.3)
ax.invert_xaxis()

ax = axes[1]
freqs = [1, 5, 21]
flabels = ["Diario", "Semanal", "Mensual"]
hrp_large = [sharpe("data_large", f, "HRP") for f in freqs]
hrp_xl = [sharpe("data_xl", f, "HRP") for f in freqs]
ew_large = [sharpe("data_large", f, "Equal_Weight") for f in freqs]
ew_xl = [sharpe("data_xl", f, "Equal_Weight") for f in freqs]

x = np.arange(len(freqs))
ax.plot(x, hrp_large, "-s", color="#C0392B", label="HRP (62 act.)", linewidth=1.8)
ax.plot(x, hrp_xl, "-o", color="#7D3C98", label="HRP (2005–24)", linewidth=1.8)
ax.plot(x, ew_large, "--", color="#D35400", alpha=0.6, label="EW (62 act.)")
ax.plot(x, ew_xl, ":", color="#7D3C98", alpha=0.5, label="EW (2005–24)")
ax.axhline(0, color="black", linewidth=0.8)
for xi, y in zip(x, hrp_large):
    ax.annotate(f"{y:.2f}", (xi, y), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8, color="#C0392B")
for xi, y in zip(x, hrp_xl):
    ax.annotate(f"{y:.2f}", (xi, y), textcoords="offset points", xytext=(0, -13),
                ha="center", fontsize=8, color="#7D3C98")
ax.set_xticks(x)
ax.set_xticklabels(flabels)
ax.set_xlabel("Frecuencia de rebalanceo")
ax.set_ylabel("Ratio de Sharpe neto")
ax.set_title("(b) La frecuencia rescata a HRP del coste", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
out = os.path.normpath(os.path.join(FIG_DIR, "figura_riesgo_rebalanceo.png"))
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figura guardada en: {out}")
