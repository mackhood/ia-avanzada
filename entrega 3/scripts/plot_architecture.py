"""Genera el diagrama conceptual de la arquitectura de la Entrega 3 (rama temporal intercambiable)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "figura_arquitectura_entrega3.png"

NAVY = "#2c3e50"
FILL = "#f5f7fa"


def box(ax, xy, w, h, text, fontsize=13, fill=FILL):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8, edgecolor=NAVY, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#111827")
    return (x, y, w, h)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color=NAVY))


fig, ax = plt.subplots(figsize=(13, 8.6))
ax.set_xlim(0, 13)
ax.set_ylim(-0.3, 8)
ax.axis("off")
ax.set_title("Arquitectura hibrida - Entrega 3 (rama temporal intercambiable)", fontsize=20, fontweight="bold", pad=18)

# Rama temporal (intercambiable)
b_ctx = box(ax, (0.4, 5.4), 3.4, 1.5, "Contexto temporal\n4 semanas x 9 features")
b_branch = box(
    ax, (4.6, 4.9), 3.6, 2.5,
    "Rama temporal\n(intercambiable)\n\nLSTM 32  /  GRU 32  /\nConv1D 32x2 + GlobalMaxPool",
    fontsize=12, fill="#eaf2ff",
)
arrow(ax, (3.8, 6.15), (4.6, 6.15))

# Rama de la cancion
b_song = box(ax, (0.4, 1.6), 3.4, 1.5, "Features cancion\n40 variables")
b_dense = box(ax, (4.6, 1.6), 3.6, 1.5, "Dense\n32 neuronas")
arrow(ax, (3.8, 2.35), (4.6, 2.35))

# Concatenacion
b_concat = box(ax, (9.0, 3.4), 3.2, 2.0, "Concatenacion")
arrow(ax, (8.2, 6.0), (9.0, 4.7))
arrow(ax, (8.2, 2.35), (9.0, 3.9))

# Salida
b_out1 = box(ax, (9.0, 1.4), 3.2, 1.3, "Dense + Dropout")
arrow(ax, (10.6, 3.4), (10.6, 2.7))
b_out2 = box(ax, (9.0, 0.0), 3.2, 1.1, "Sigmoide\nP(hit)", fill="#eafaf1")
arrow(ax, (9.8, 1.4), (9.4, 1.1))

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Wrote {OUT_PATH}")
