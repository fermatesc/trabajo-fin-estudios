"""
Genera la figura comparativa del experimento de escalado (20 vs 62 activos)
para el Capítulo 5 del TFM. Lee evaluate_fix.json (20 activos) y
evaluate_scaled.json (62 activos) y produce un panel doble:
  (a) Sharpe neto por configuración
  (b) Turnover medio diario por configuración
Salida: figuras/ del proyecto LaTeX.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, "..", "..", "Plantilla LATEX TFE TFM UNIR", "figuras")

with open(os.path.join(REPO, "evaluate_fix.json")) as f:
    d20 = json.load(f)
with open(os.path.join(REPO, "evaluate_scaled.json")) as f:
    d62 = json.load(f)

labels = ["Equal-Weight", "Mitigada\n($\\tau_s$=1.5 + penaliz.)", "Base\n($\\tau_s$=5.0)"]

sharpe_20 = [d20["Equal_Weight"]["sharpe_neto"], d20["FIXED"]["sharpe_mean"], d20["BASELINE"]["sharpe_mean"]]
sharpe_62 = [d62["Equal_Weight"]["sharpe_neto"], d62["FIXED"]["sharpe_mean"], d62["BASELINE"]["sharpe_mean"]]
err_20 = [0.0, d20["FIXED"]["sharpe_std"], d20["BASELINE"]["sharpe_std"]]
err_62 = [0.0, d62["FIXED"]["sharpe_std"], d62["BASELINE"]["sharpe_std"]]

turn_20 = [d20["Equal_Weight"]["turnover_medio_diario"], d20["FIXED"]["turnover_medio_diario"], d20["BASELINE"]["turnover_medio_diario"]]
turn_62 = [d62["Equal_Weight"]["turnover_medio_diario"], d62["FIXED"]["turnover_medio_diario"], d62["BASELINE"]["turnover_medio_diario"]]

x = np.arange(len(labels))
w = 0.38
C20, C62 = "#9E9E9E", "#1F4E79"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

ax1.bar(x - w/2, sharpe_20, w, yerr=err_20, capsize=3, color=C20, label="20 activos (5 épocas)", edgecolor="black", linewidth=0.4)
ax1.bar(x + w/2, sharpe_62, w, yerr=err_62, capsize=3, color=C62, label="62 activos (30 épocas)", edgecolor="black", linewidth=0.4)
ax1.axhline(0, color="black", linewidth=0.6)
ax1.set_ylabel("Ratio de Sharpe neto")
ax1.set_title("(a) Sharpe neto (con costes)", fontsize=11)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
ax1.grid(axis="y", alpha=0.3)
ax1.legend(fontsize=8.5, loc="upper right")
for xi, v20, v62 in zip(x, sharpe_20, sharpe_62):
    ax1.text(xi - w/2, v20 + (0.04 if v20 >= 0 else -0.10), f"{v20:.2f}", ha="center", fontsize=8)
    ax1.text(xi + w/2, v62 + (0.04 if v62 >= 0 else -0.10), f"{v62:.2f}", ha="center", fontsize=8)

ax2.bar(x - w/2, turn_20, w, color=C20, edgecolor="black", linewidth=0.4)
ax2.bar(x + w/2, turn_62, w, color=C62, edgecolor="black", linewidth=0.4)
ax2.set_ylabel("Turnover medio diario")
ax2.set_title("(b) Rotación diaria de la cartera", fontsize=11)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
ax2.grid(axis="y", alpha=0.3)
for xi, v20, v62 in zip(x, turn_20, turn_62):
    ax2.text(xi - w/2, v20 + 0.012, f"{v20:.3f}", ha="center", fontsize=8)
    ax2.text(xi + w/2, v62 + 0.012, f"{v62:.3f}", ha="center", fontsize=8)

plt.tight_layout()
out = os.path.normpath(os.path.join(FIG_DIR, "figura_escalado_activos.png"))
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Figura guardada en: {out}")
