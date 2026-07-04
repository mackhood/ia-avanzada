"""
Entrega 3 - Entrenamiento y comparacion de arquitecturas (LSTM / GRU / Conv1D)
sobre el dataset deduplicado (una fila por cancion).

Cambios respecto de la Entrega 2:
  1. Dataset sin informacion duplicada (build_dedup_dataset.py): cada cancion
     aparece una unica vez, en lugar de una fila por cada semana que
     permanecio en el ranking.
  2. Split estratificado por clase en lugar de puramente cronologico. Al
     deduplicar, las canciones hit quedaron muy concentradas en la primera
     semana utilizable del dataset (la mayoria ya venia siendo un hit antes
     de que empezara la ventana de datos observada). Un corte cronologico
     puro dejaba 1 hit en validacion y 2 en test, con lo cual F1/precision/
     recall en esos conjuntos no son estadisticamente interpretables. Se
     mantiene la semilla fija y el criterio de que cada cancion aparece en
     un unico split (ya garantizado por la deduplicacion).
  3. Comparacion de tres ramas temporales para el contexto de 4 semanas
     previas: LSTM (linea base de la Entrega 2), GRU y Conv1D, manteniendo
     el resto de la arquitectura (rama densa de la cancion, concatenacion,
     cabeza de clasificacion) sin cambios para poder aislar el efecto de la
     capa temporal.

Salida por arquitectura (dentro de Entrega 3/resultados_<arch>/):
  modelo_<arch>_hits.keras
  context_scaler_<arch>.joblib
  song_preprocessor_<arch>.joblib
  best_threshold_<arch>.joblib
  metricas_<arch>.txt
  predicciones_test_<arch>.csv
  figura_matriz_confusion_<arch>.png
  figura_metricas_<arch>.png

Ademas genera comparacion_arquitecturas.csv y comparacion_arquitecturas.md
con la comparacion conjunta de las tres arquitecturas.
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
from sklearn.model_selection import train_test_split
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
DATASET_PATH = BASE_DIR / "tracks_charts_dedup_dataset.csv"

WINDOW_SIZE = 4
CONTEXT_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness",
]
CONTEXT_COLS_FLAT = [
    f"ctx_{feature}_t_minus_{lag}"
    for lag in range(WINDOW_SIZE, 0, -1)
    for feature in CONTEXT_FEATURES
]

SONG_NUMERIC_FEATURES = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "release_year",
    "delta_danceability_vs_prev4_hit_avg",
    "delta_energy_vs_prev4_hit_avg",
    "delta_speechiness_vs_prev4_hit_avg",
    "delta_acousticness_vs_prev4_hit_avg",
    "delta_instrumentalness_vs_prev4_hit_avg",
    "delta_liveness_vs_prev4_hit_avg",
    "delta_valence_vs_prev4_hit_avg",
    "delta_tempo_vs_prev4_hit_avg",
    "delta_loudness_vs_prev4_hit_avg",
]
SONG_CATEGORICAL_FEATURES = ["key", "mode", "time_signature"]
TARGET_COL = "hit"

ARCHITECTURES = ["lstm", "gru", "conv1d"]


def build_temporal_branch(arch_name: str, context_input):
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
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    ]
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
    out_dir = BASE_DIR / f"resultados_{arch_name}"
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

    print(f"\n{'=' * 70}\nArquitectura: {arch_name.upper()}\n{'=' * 70}")
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
        f"Matriz de confusion - test ({arch_name.upper()})",
    )
    plot_metrics_bar(
        test_metrics,
        out_dir / f"figura_metricas_{arch_name}.png",
        f"Metricas en test - arquitectura {arch_name.upper()}",
    )

    with open(out_dir / f"metricas_{arch_name}.txt", "w", encoding="utf-8") as f:
        f.write(f"Modelo {arch_name.upper()} - metricas (Entrega 3, dataset deduplicado)\n")
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

    # La plantilla de entregas exige un minimo de 25% de los ejemplos para
    # prueba; se usa 60% train / 15% validation / 25% test.
    train_val_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15 / 0.75, stratify=train_val_df[TARGET_COL], random_state=RANDOM_STATE
    )

    print("Train:", train_df.shape, "hits:", int(train_df["hit"].sum()))
    print("Val:", val_df.shape, "hits:", int(val_df["hit"].sum()))
    print("Test:", test_df.shape, "hits:", int(test_df["hit"].sum()))

    results = []
    for arch_name in ARCHITECTURES:
        result = train_architecture(arch_name, df, train_df, val_df, test_df)
        results.append(result)

    comparison_rows = []
    for result in results:
        test_metrics = result["test"]
        comparison_rows.append(
            {
                "arquitectura": result["architecture"].upper(),
                "umbral": result["best_threshold"],
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics["roc_auc"],
                "pr_auc": test_metrics["pr_auc"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(BASE_DIR / "comparacion_arquitecturas.csv", index=False)

    with open(BASE_DIR / "comparacion_arquitecturas.md", "w", encoding="utf-8") as f:
        f.write("# Comparacion de arquitecturas - conjunto de prueba (Entrega 3)\n\n")
        f.write("| Arquitectura | Umbral | Accuracy | Precision | Recall | F1-score | ROC AUC | PR AUC |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in comparison_rows:
            f.write(
                f"| {row['arquitectura']} | {row['umbral']:.2f} | {row['accuracy']:.2%} | "
                f"{row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} | "
                f"{row['roc_auc']:.2%} | {row['pr_auc']:.2%} |\n"
            )

    with open(BASE_DIR / "resultados_completos.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nComparacion final:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
