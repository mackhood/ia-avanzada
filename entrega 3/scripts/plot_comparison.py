"""Genera el grafico comparativo Entrega 2 vs Entrega 3 (arquitecturas) para el informe."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# Metricas de test reportadas en la Entrega 2 (dataset con duplicacion por semana).
entrega2 = {
    "accuracy": 0.9881,
    "precision": 0.9455,
    "recall": 1.0000,
    "f1": 0.9720,
    "roc_auc": 0.9969,
}

comparison = pd.read_csv(BASE_DIR / "comparacion_arquitecturas_grouped.csv")

metrics_labels = ["accuracy", "precision", "recall", "f1", "roc_auc"]
display_labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]

lstm = comparison[comparison["arquitectura"] == "LSTM"].iloc[0]
gru = comparison[comparison["arquitectura"] == "GRU"].iloc[0]
conv1d = comparison[comparison["arquitectura"] == "CONV1D"].iloc[0]

series = {
    "Entrega 2\n(LSTM, split cronologico con fuga)": [entrega2[m] for m in metrics_labels],
    "Entrega 3 - LSTM\n(split por grupo de cancion)": [lstm[m] for m in metrics_labels],
    "Entrega 3 - GRU\n(split por grupo de cancion)": [gru[m] for m in metrics_labels],
    "Entrega 3 - Conv1D\n(split por grupo de cancion)": [conv1d[m] for m in metrics_labels],
}

x = np.arange(len(display_labels))
width = 0.2
colors = ["#94A3B8", "#4C72B0", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(10, 5.5))
for i, (label, values) in enumerate(series.items()):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, values, width, label=label, color=colors[i])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.0%}", ha="center", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(display_labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Valor")
ax.set_title("Comparacion de metricas en test: Entrega 2 vs Entrega 3 (split por grupo)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(BASE_DIR / "figura_comparacion_entrega2_vs_entrega3.png", dpi=150)
print("Wrote figura_comparacion_entrega2_vs_entrega3.png")
