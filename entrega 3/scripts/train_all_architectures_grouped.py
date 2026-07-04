"""
Entrega 3 (version final) - Entrenamiento y comparacion de arquitecturas
(LSTM / GRU / Conv1D) sobre el dataset a nivel semana, con split por grupo de
cancion.

Iteracion previa (train_all_architectures.py + build_dedup_dataset.py): se
habia probado eliminar toda repeticion de canciones, dejando una unica fila
por cancion. Al revisarlo con mas detalle se detecto que ese enfoque tiraba
una señal real del fenomeno (que un hit permanece varias semanas en el
ranking) y reducia demasiado el tamaño muestral (de 769 a 79 hits).

El problema real senalado por el docente no es que una cancion aparezca en
mas de una fila (cada semana tiene un contexto de tendencias distinto, no es
una copia identica), sino que el corte train/val/test de la Entrega 2 era
cronologico a nivel de fila: una misma cancion podia terminar con semanas en
entrenamiento y semanas en prueba, filtrando su "fingerprint" de audio entre
ambos conjuntos.

Esta version final resuelve el problema en el split, no en el dataset:
  1. Se usa el dataset completo a nivel de cancion x semana
     (tracks_charts_grouped_dataset.csv, 2019 filas, 769 hits), construido por
     build_grouped_dataset.py.
  2. El split train/val/test se hace por GRUPO de cancion (clean_name +
     clean_artists): todas las semanas de una misma cancion quedan en un unico
     subconjunto. Se usa un algoritmo voraz (greedy bin-packing) que asigna
     canciones completas a train/val/test buscando aproximar las proporciones
     60% / 15% / 25% de FILAS en cada clase, ya que el tamaño de cada cancion
     (cantidad de semanas en el ranking) es muy variable.
  3. Se comparan las mismas tres arquitecturas de la iteracion anterior (LSTM,
     GRU, Conv1D) para la rama temporal, bajo esta metodologia corregida.

Salida por arquitectura (dentro de Entrega 3/resultados_<arch>_grouped/):
  modelo_<arch>_hits.keras, preprocesadores, metricas, predicciones, figuras.

Ademas genera comparacion_arquitecturas_grouped.csv/.md y
split_asignacion_canciones.csv (que cancion quedo en que subconjunto).
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "tracks_charts_grouped_dataset.csv"
GROUP_COLS = ["clean_name", "clean_artists"]
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.15, "test": 0.25}

WINDOW_SIZE = 4
CONTEXT_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "loudness",
]
CONTEXT_COLS_FLAT = [
    f"ctx_{feature}_t_minus_{lag}"
    for lag in range(WINDOW_SIZE, 0, -1)
    for feature in CONTEXT_FEATURES
]

SONG_NUMERIC_FEATURES = [
    "duration_ms", "explicit", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "release_year",
    "delta_danceability_vs_prev4_hit_avg", "delta_energy_vs_prev4_hit_avg",
    "delta_speechiness_vs_prev4_hit_avg", "delta_acousticness_vs_prev4_hit_avg",
    "delta_instrumentalness_vs_prev4_hit_avg", "delta_liveness_vs_prev4_hit_avg",
    "delta_valence_vs_prev4_hit_avg", "delta_tempo_vs_prev4_hit_avg",
    "delta_loudness_vs_prev4_hit_avg",
]
SONG_CATEGORICAL_FEATURES = ["key", "mode", "time_signature"]
TARGET_COL = "hit"

ARCHITECTURES = ["lstm", "gru", "conv1d"]


def greedy_group_split(df, group_cols, target_col, fractions, seed):
    """Asigna cada grupo (cancion) completo a un unico split, buscando
    aproximar las proporciones de FILAS pedidas dentro de cada clase por
    separado (bin-packing voraz: en cada paso, el grupo se asigna al split
    con mayor deficit respecto de su objetivo)."""

    rng = np.random.default_rng(seed)
    split_names = list(fractions.keys())
    assignment = {}

    for _, class_df in df.groupby(target_col):
        groups = class_df.groupby(group_cols).size().reset_index(name="count")
        order = rng.permutation(len(groups))
        groups = groups.iloc[order].sort_values("count", ascending=False, kind="stable")

        total = int(groups["count"].sum())
        assigned = {name: 0 for name in split_names}
        for _, grow in groups.iterrows():
            key = tuple(grow[c] for c in group_cols)
            deficits = {name: fractions[name] * total - assigned[name] for name in split_names}
            choice = max(deficits, key=deficits.get)
            assigned[choice] += int(grow["count"])
            assignment[key] = choice

    return assignment


def build_temporal_branch(arch_name, context_input):
    if arch_name == "lstm":
        return LSTM(32, name="lstm_context")(context_input)
    if arch_name == "gru":
        return GRU(32, name="gru_context")(context_input)
    if arch_name == "conv1d":
        x = Conv1D(32, kernel_size=2, activation="relu", padding="causal", name="conv1d_context")(context_input)
        x = Conv1D(32, kernel_size=2, activation="relu", padding="causal", name="conv1d_context_2")(x)
        return GlobalMaxPooling1D(name="global_max_pool_context")(x)
    raise ValueError(f"Unknown architecture: {arch_name}")


def make_lstm_input(dataframe, context_scaler):
    context_2d = context_scaler.transform(dataframe[CONTEXT_COLS_FLAT])
    return context_2d.reshape(len(dataframe), WINDOW_SIZE, len(CONTEXT_FEATURES))


def evaluate_split(model, name, X_context, X_song, y_true, threshold=0.5):
    y_proba = model.predict([X_context, X_song], verbose=0).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "split": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    return y_proba, y_pred, metrics


def print_metrics(metrics):
    print(f"\n{metrics['split']} (n={metrics['n']}, positivos={metrics['n_positive']})")
    print("Threshold:", metrics["threshold"])
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC AUC:", round(metrics["roc_auc"], 4))
    print("PR AUC:", round(metrics["pr_auc"], 4))
    print("Confusion matrix:", metrics["confusion_matrix"])


def plot_confusion_matrix(cm, out_path, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No hit", "Hit"])
    ax.set_yticklabels(["No hit", "Hit"])
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics_bar(metrics, out_path, title):
    labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC", "PR AUC"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"],
              metrics["f1"], metrics["roc_auc"], metrics["pr_auc"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title(title)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_architecture(arch_name, df, train_df, val_df, test_df):
    out_dir = BASE_DIR / f"resultados_{arch_name}_grouped"
    out_dir.mkdir(exist_ok=True)

    context_scaler = StandardScaler()
    context_scaler.fit(train_df[CONTEXT_COLS_FLAT])

    X_context_train = make_lstm_input(train_df, context_scaler)
    X_context_val = make_lstm_input(val_df, context_scaler)
    X_context_test = make_lstm_input(test_df, context_scaler)

    try:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    song_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SONG_NUMERIC_FEATURES),
            ("cat", categorical_encoder, SONG_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    X_song_train = song_preprocessor.fit_transform(train_df)
    X_song_val = song_preprocessor.transform(val_df)
    X_song_test = song_preprocessor.transform(test_df)

    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    y_val = val_df[TARGET_COL].astype(int).to_numpy()
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    class_counts = np.bincount(y_train)
    total = class_counts.sum()
    class_weight = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}

    tf.random.set_seed(RANDOM_STATE)
    context_input = Input(shape=(WINDOW_SIZE, len(CONTEXT_FEATURES)), name="temporal_context")
    x_context = build_temporal_branch(arch_name, context_input)
    x_context = Dropout(0.25)(x_context)

    song_input = Input(shape=(X_song_train.shape[1],), name="song_features")
    x_song = Dense(32, activation="relu")(song_input)
    x_song = Dropout(0.25)(x_song)

    x = Concatenate()([x_context, x_song])
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.25)(x)
    output = Dense(1, activation="sigmoid", name="hit_probability")(x)

    model = Model(inputs=[context_input, song_input], outputs=output, name=f"hit_prediction_{arch_name}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    print(f"\n{'=' * 70}\nArquitectura: {arch_name.upper()} (split por grupo)\n{'=' * 70}")
    model.summary()

    early_stopping = EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True)

    model.fit(
        [X_context_train, X_song_train],
        y_train,
        validation_data=([X_context_val, X_song_val], y_val),
        epochs=150,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0,
    )

    val_proba, _, val_metrics_default = evaluate_split(model, "Validation", X_context_val, X_song_val, y_val)
    print_metrics(val_metrics_default)

    thresholds = np.arange(0.05, 0.96, 0.01)
    f1_by_threshold = [f1_score(y_val, (val_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1_by_threshold))])
    print("Best validation threshold:", round(best_threshold, 2))

    test_proba, test_pred, test_metrics = evaluate_split(
        model, "Test", X_context_test, X_song_test, y_test, best_threshold
    )
    print_metrics(test_metrics)

    test_results = test_df[["date", "song", "artists", "hit"]].copy()
    test_results["predicted_probability"] = test_proba
    test_results["predicted_hit"] = test_pred
    test_results["correct"] = test_results["hit"] == test_results["predicted_hit"]

    model.save(out_dir / f"modelo_{arch_name}_hits.keras")
    joblib.dump(context_scaler, out_dir / f"context_scaler_{arch_name}.joblib")
    joblib.dump(song_preprocessor, out_dir / f"song_preprocessor_{arch_name}.joblib")
    joblib.dump(best_threshold, out_dir / f"best_threshold_{arch_name}.joblib")
    test_results.to_csv(out_dir / f"predicciones_test_{arch_name}.csv", index=False)

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        out_dir / f"figura_matriz_confusion_{arch_name}.png",
        f"Matriz de confusion - test ({arch_name.upper()}, split por grupo)",
    )
    plot_metrics_bar(
        test_metrics,
        out_dir / f"figura_metricas_{arch_name}.png",
        f"Metricas en test - {arch_name.upper()} (split por grupo)",
    )

    with open(out_dir / f"metricas_{arch_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Modelo {arch_name.upper()} - metricas (Entrega 3, split por grupo de cancion)\n")
        f.write(f"Dataset: {DATASET_PATH.name}\n")
        f.write(f"Filas totales: {df.shape[0]} (hit={int(df['hit'].sum())}, no-hit={int((df['hit']==0).sum())})\n")
        f.write(f"Train: {train_df.shape[0]} filas ({int(y_train.sum())} hit)\n")
        f.write(f"Validation: {val_df.shape[0]} filas ({int(y_val.sum())} hit)\n")
        f.write(f"Test: {test_df.shape[0]} filas ({int(y_test.sum())} hit)\n")
        f.write(f"Mejor umbral por F1 en validacion: {best_threshold:.2f}\n\n")
        for metrics in [val_metrics_default, test_metrics]:
            f.write(f"{metrics['split']}\n")
            f.write(f"Threshold: {metrics['threshold']:.4f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1: {metrics['f1']:.4f}\n")
            f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
            f.write(f"Confusion matrix: {metrics['confusion_matrix']}\n")
            f.write(metrics["classification_report"])
            f.write("\n")

    return {"architecture": arch_name, "val": val_metrics_default, "test": test_metrics, "best_threshold": best_threshold}


def main():
    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print("Dataset shape:", df.shape)
    print(df["hit"].value_counts().sort_index())
    print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
    print("Unique songs:", df.groupby(GROUP_COLS).ngroups)

    assignment = greedy_group_split(df, GROUP_COLS, TARGET_COL, SPLIT_FRACTIONS, RANDOM_STATE)
    split_series = df.apply(lambda row: assignment[tuple(row[c] for c in GROUP_COLS)], axis=1)
    df["split"] = split_series

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Verificacion: ninguna cancion debe aparecer en mas de un split.
    songs_by_split = {name: set(map(tuple, sub[GROUP_COLS].drop_duplicates().to_numpy()))
                       for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]}
    assert not (songs_by_split["train"] & songs_by_split["val"])
    assert not (songs_by_split["train"] & songs_by_split["test"])
    assert not (songs_by_split["val"] & songs_by_split["test"])

    print("Train:", train_df.shape, "hits:", int(train_df["hit"].sum()),
          "canciones:", len(songs_by_split["train"]))
    print("Val:", val_df.shape, "hits:", int(val_df["hit"].sum()),
          "canciones:", len(songs_by_split["val"]))
    print("Test:", test_df.shape, "hits:", int(test_df["hit"].sum()),
          "canciones:", len(songs_by_split["test"]))

    df[GROUP_COLS + ["hit", "split"]].drop_duplicates(subset=GROUP_COLS).to_csv(
        BASE_DIR / "split_asignacion_canciones.csv", index=False
    )

    results = []
    for arch_name in ARCHITECTURES:
        result = train_architecture(arch_name, df, train_df, val_df, test_df)
        results.append(result)

    comparison_rows = []
    for result in results:
        test_metrics = result["test"]
        comparison_rows.append({
            "arquitectura": result["architecture"].upper(),
            "umbral": result["best_threshold"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(BASE_DIR / "comparacion_arquitecturas_grouped.csv", index=False)

    with open(BASE_DIR / "comparacion_arquitecturas_grouped.md", "w", encoding="utf-8") as f:
        f.write("# Comparacion de arquitecturas - conjunto de prueba (Entrega 3, split por grupo)\n\n")
        f.write("| Arquitectura | Umbral | Accuracy | Precision | Recall | F1-score | ROC AUC | PR AUC |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in comparison_rows:
            f.write(
                f"| {row['arquitectura']} | {row['umbral']:.2f} | {row['accuracy']:.2%} | "
                f"{row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} | "
                f"{row['roc_auc']:.2%} | {row['pr_auc']:.2%} |\n"
            )

    with open(BASE_DIR / "resultados_completos_grouped.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nComparacion final (split por grupo):")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
