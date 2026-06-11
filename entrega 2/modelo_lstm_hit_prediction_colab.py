# Prototipo LSTM - Prediccion de hits musicales
# Trabajo Practico IA Avanzada - Grupo B
#
# Este archivo esta pensado para ejecutarse en Google Colab.
# Subir al entorno el archivo:
#   tracks_charts_classification_dataset.csv

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
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
from tensorflow.keras.layers import Concatenate, Dense, Dropout, LSTM


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


DATASET_PATH = "tracks_charts_classification_dataset.csv"
MODEL_PATH = "modelo_lstm_hits.keras"
CONTEXT_SCALER_PATH = "context_scaler.joblib"
SONG_PREPROCESSOR_PATH = "song_preprocessor.joblib"
THRESHOLD_PATH = "best_threshold.joblib"
METRICS_PATH = "modelo_lstm_metricas.txt"
TEST_PREDICTIONS_PATH = "modelo_lstm_predicciones_test.csv"

df = pd.read_csv(DATASET_PATH)
df["date"] = pd.to_datetime(df["date"])

print("Dataset shape:", df.shape)
print(df["hit"].value_counts().sort_index())
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())


# Columnas de contexto temporal.
# Cada muestra tiene 4 semanas previas y 9 variables por semana.
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
WINDOW_SIZE = 4

context_cols_by_step = [
    [f"ctx_{feature}_t_minus_{lag}" for feature in CONTEXT_FEATURES]
    for lag in range(WINDOW_SIZE, 0, -1)
]
context_cols_flat = [col for step_cols in context_cols_by_step for col in step_cols]

song_numeric_features = [
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

song_categorical_features = [
    "key",
    "mode",
    "time_signature",
]

target_col = "hit"


# Split temporal: evita entrenar con semanas posteriores para predecir semanas anteriores.
weeks = np.array(sorted(df["date"].unique()))
n_weeks = len(weeks)
train_end = int(n_weeks * 0.70)
val_end = int(n_weeks * 0.85)

train_weeks = weeks[:train_end]
val_weeks = weeks[train_end:val_end]
test_weeks = weeks[val_end:]

train_df = df[df["date"].isin(train_weeks)].copy()
val_df = df[df["date"].isin(val_weeks)].copy()
test_df = df[df["date"].isin(test_weeks)].copy()

print("Train weeks:", pd.Timestamp(train_weeks[0]).date(), "to", pd.Timestamp(train_weeks[-1]).date(), train_df.shape)
print("Val weeks:", pd.Timestamp(val_weeks[0]).date(), "to", pd.Timestamp(val_weeks[-1]).date(), val_df.shape)
print("Test weeks:", pd.Timestamp(test_weeks[0]).date(), "to", pd.Timestamp(test_weeks[-1]).date(), test_df.shape)


# Escalado del contexto temporal.
# Se ajusta solo con train para evitar fuga de informacion.
context_scaler = StandardScaler()
context_scaler.fit(train_df[context_cols_flat])


def make_lstm_input(dataframe: pd.DataFrame) -> np.ndarray:
    context_2d = context_scaler.transform(dataframe[context_cols_flat])
    return context_2d.reshape(len(dataframe), WINDOW_SIZE, len(CONTEXT_FEATURES))


X_context_train = make_lstm_input(train_df)
X_context_val = make_lstm_input(val_df)
X_context_test = make_lstm_input(test_df)


# Preprocesamiento de caracteristicas propias de la cancion.
try:
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

song_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), song_numeric_features),
        ("cat", categorical_encoder, song_categorical_features),
    ],
    remainder="drop",
)

X_song_train = song_preprocessor.fit_transform(train_df)
X_song_val = song_preprocessor.transform(val_df)
X_song_test = song_preprocessor.transform(test_df)

y_train = train_df[target_col].astype(int).to_numpy()
y_val = val_df[target_col].astype(int).to_numpy()
y_test = test_df[target_col].astype(int).to_numpy()

print("LSTM input:", X_context_train.shape)
print("Song input:", X_song_train.shape)


# Pesos de clase para compensar el desbalance moderado.
class_counts = np.bincount(y_train)
total = class_counts.sum()
class_weight = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1]),
}
print("Class weights:", class_weight)


# Arquitectura:
# - Entrada 1: secuencia de contexto temporal de las ultimas 4 semanas.
# - Entrada 2: atributos de la cancion candidata.
# - Salida: probabilidad de hit.
context_input = Input(shape=(WINDOW_SIZE, len(CONTEXT_FEATURES)), name="temporal_context")
x_context = LSTM(32, name="lstm_context")(context_input)
x_context = Dropout(0.25)(x_context)

song_input = Input(shape=(X_song_train.shape[1],), name="song_features")
x_song = Dense(32, activation="relu")(song_input)
x_song = Dropout(0.25)(x_song)

x = Concatenate()([x_context, x_song])
x = Dense(32, activation="relu")(x)
x = Dropout(0.25)(x)
output = Dense(1, activation="sigmoid", name="hit_probability")(x)

model = Model(inputs=[context_input, song_input], outputs=output)
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

model.summary()


early_stopping = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=12,
    restore_best_weights=True,
)

history = model.fit(
    [X_context_train, X_song_train],
    y_train,
    validation_data=([X_context_val, X_song_val], y_val),
    epochs=120,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stopping],
    verbose=1,
)


def evaluate_split(name: str, X_context: np.ndarray, X_song: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    y_proba = model.predict([X_context, X_song]).ravel()
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "split": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }

    print(f"\n{name}")
    print("Threshold:", threshold)
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC AUC:", round(metrics["roc_auc"], 4))
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])

    return y_proba, y_pred, metrics


val_proba, _, val_metrics_default = evaluate_split("Validation", X_context_val, X_song_val, y_val)


# Busqueda simple de umbral usando validacion para maximizar F1.
thresholds = np.arange(0.10, 0.91, 0.01)
f1_by_threshold = []
for threshold in thresholds:
    val_pred = (val_proba >= threshold).astype(int)
    f1_by_threshold.append(f1_score(y_val, val_pred, zero_division=0))

best_threshold = float(thresholds[int(np.argmax(f1_by_threshold))])
print("Best validation threshold:", round(best_threshold, 2))

test_proba, test_pred, test_metrics = evaluate_split("Test", X_context_test, X_song_test, y_test, best_threshold)


# Ejemplos de predicciones en test para documentar casos exitosos y fallidos.
test_results = test_df[
    [
        "date",
        "song",
        "artists",
        "hit",
        "danceability",
        "energy",
        "valence",
        "tempo",
    ]
].copy()
test_results["predicted_probability"] = test_proba
test_results["predicted_hit"] = test_pred
test_results["correct"] = test_results["hit"] == test_results["predicted_hit"]

print("\nTop predicted hits in test:")
print(test_results.sort_values("predicted_probability", ascending=False).head(15).to_string(index=False))

print("\nMisclassified examples in test:")
print(test_results[test_results["correct"] == False].sort_values("predicted_probability", ascending=False).head(15).to_string(index=False))


model.save(MODEL_PATH)
joblib.dump(context_scaler, CONTEXT_SCALER_PATH)
joblib.dump(song_preprocessor, SONG_PREPROCESSOR_PATH)
joblib.dump(best_threshold, THRESHOLD_PATH)
test_results.to_csv(TEST_PREDICTIONS_PATH, index=False)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    f.write("Modelo LSTM - metricas del prototipo\n")
    f.write(f"Dataset: {DATASET_PATH}\n")
    f.write(f"Filas: {df.shape[0]}, columnas: {df.shape[1]}\n")
    f.write(f"Rango de fechas: {df['date'].min().date()} a {df['date'].max().date()}\n")
    f.write(f"Train: {pd.Timestamp(train_weeks[0]).date()} a {pd.Timestamp(train_weeks[-1]).date()} - {train_df.shape[0]} filas\n")
    f.write(f"Validation: {pd.Timestamp(val_weeks[0]).date()} a {pd.Timestamp(val_weeks[-1]).date()} - {val_df.shape[0]} filas\n")
    f.write(f"Test: {pd.Timestamp(test_weeks[0]).date()} a {pd.Timestamp(test_weeks[-1]).date()} - {test_df.shape[0]} filas\n")
    f.write(f"Mejor umbral por F1 en validacion: {best_threshold:.2f}\n\n")
    for metrics in [val_metrics_default, test_metrics]:
        f.write(f"{metrics['split']}\n")
        f.write(f"Threshold: {metrics['threshold']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write("Confusion matrix:\n")
        f.write(str(metrics["confusion_matrix"]))
        f.write("\n")
        f.write(metrics["classification_report"])
        f.write("\n")

print("\nSaved artifacts:")
print(MODEL_PATH)
print(CONTEXT_SCALER_PATH)
print(SONG_PREPROCESSOR_PATH)
print(THRESHOLD_PATH)
print(METRICS_PATH)
print(TEST_PREDICTIONS_PATH)
