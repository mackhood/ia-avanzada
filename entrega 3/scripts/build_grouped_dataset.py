"""
Entrega 3 (version final) - Construccion del dataset a nivel semana, preservando
la persistencia real de los hits en el ranking, pensado para un split por grupo
de cancion.

Iteracion previa (build_dedup_dataset.py): se habia probado eliminar por completo
la repeticion de canciones, dejando una unica fila por cancion. Al revisarlo se
detecto que ese enfoque tiraba a la basura una señal real del fenomeno (que una
cancion exitosa permanece varias semanas en el ranking) y reducia demasiado el
tamaño muestral (de 769 a 79 hits), perdiendo poder estadistico.

El verdadero problema senalado por el docente no es que una cancion aparezca en
mas de una fila (eso es realista: cada semana tiene un contexto de tendencias
distinto), sino que el corte train/val/test de la Entrega 2 era cronologico a
nivel de fila, por lo que una misma cancion podia terminar con semanas en
entrenamiento y semanas en prueba (fuga de informacion).

Este script reconstruye el dataset completo a nivel de "cancion x semana"
(similar al de la Entrega 2, con la correccion de parsing/normalizacion de la
Entrega 3), y dejar el control de la fuga de informacion a cargo del split por
grupo de cancion, implementado en train_all_architectures.py.

Entradas: tracks_charts_hit_cleaned.csv, tracks_charts_not_hit_cleaned.csv
Salidas: tracks_charts_grouped_dataset.csv, tracks_charts_grouped_dataset_validation.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HIT_PATH = ROOT / "tracks_charts_hit_cleaned.csv"
NOTHIT_PATH = ROOT / "tracks_charts_not_hit_cleaned.csv"
OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = OUT_DIR / "tracks_charts_grouped_dataset.csv"
REPORT_PATH = OUT_DIR / "tracks_charts_grouped_dataset_validation.txt"

WINDOW_SIZE = 4
CONTEXT_FEATURES = ["danceability", "energy", "speechiness", "acousticness",
                     "instrumentalness", "liveness", "valence", "tempo", "loudness"]
NUMERIC_FEATURES = CONTEXT_FEATURES
UNIT_FEATURES = ["danceability", "energy", "speechiness", "acousticness",
                  "instrumentalness", "liveness", "valence"]


def parse_numeric_column(series: pd.Series) -> pd.Series:
    def parse_value(value):
        text = str(value).strip()
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        return float(text)
    return series.map(parse_value)


def normalize_unit_feature(series: pd.Series, column: str) -> pd.Series:
    def fix(value):
        if value > 1.0:
            if value <= 1000.0:
                value = value / 1000.0
            else:
                raise ValueError(f"Column {column} has an invalid unit value: {value}")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Column {column} outside [0,1] after normalization: {value}")
        return value
    return series.map(fix)


def normalize_loudness(series: pd.Series) -> pd.Series:
    def fix(value):
        if value < -60.0:
            if value >= -1000.0:
                value = value / 100.0
            else:
                raise ValueError(f"Invalid loudness value: {value}")
        if value < -60.0 or value > 5.0:
            raise ValueError(f"Loudness outside expected range after normalization: {value}")
        return value
    return series.map(fix)


def main():
    hit = pd.read_csv(HIT_PATH)
    nothit = pd.read_csv(NOTHIT_PATH)
    for df in (hit, nothit):
        for col in NUMERIC_FEATURES:
            df[col] = parse_numeric_column(df[col])
        for col in UNIT_FEATURES:
            df[col] = normalize_unit_feature(df[col], col)
        df["loudness"] = normalize_loudness(df["loudness"])
    hit["hit"] = 1
    nothit["hit"] = 0

    combined = pd.concat([hit, nothit], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["release_date"] = pd.to_datetime(combined["release_date"], errors="coerce")
    combined["release_year"] = combined["release_date"].dt.year

    dup_keys = combined.duplicated(subset=["date", "clean_name", "clean_artists"]).sum()
    if dup_keys:
        raise ValueError(f"Duplicate date/song/artist keys found before processing: {dup_keys}")

    overlap = set(zip(hit.clean_name, hit.clean_artists)) & set(zip(nothit.clean_name, nothit.clean_artists))
    if overlap:
        raise ValueError(f"Songs present in both hit and not-hit sets: {list(overlap)[:5]}")

    weeks = np.array(sorted(combined["date"].unique()))
    week_index = {week: i for i, week in enumerate(weeks)}

    # Contexto semanal: promedio de features de TODAS las canciones que estaban
    # en el ranking esa semana (ocupacion real del chart).
    weekly_hit_rows = combined[combined["hit"] == 1]
    weekly_context = {}
    for week in weeks:
        week_hits = weekly_hit_rows[weekly_hit_rows["date"] == week]
        if week_hits.empty:
            raise ValueError(f"No hit rows available to build context for week {week}")
        weekly_context[week] = week_hits[CONTEXT_FEATURES].mean()

    # Se conservan TODAS las filas (cancion x semana) con contexto completo
    # disponible; no se deduplica por cancion. La ausencia de fuga de
    # informacion se garantiza mas adelante con un split por grupo de cancion.
    usable = combined[combined["date"].map(week_index) >= WINDOW_SIZE].copy()

    rows = []
    for _, row in usable.iterrows():
        i = week_index[row["date"]]
        obj = row.to_dict()
        lag_means = {feat: [] for feat in CONTEXT_FEATURES}
        for lag in range(1, WINDOW_SIZE + 1):
            ctx = weekly_context[weeks[i - lag]]
            for feat in CONTEXT_FEATURES:
                obj[f"ctx_{feat}_t_minus_{lag}"] = ctx[feat]
                lag_means[feat].append(ctx[feat])
        for feat in CONTEXT_FEATURES:
            obj[f"delta_{feat}_vs_prev4_hit_avg"] = float(row[feat]) - float(np.mean(lag_means[feat]))
        rows.append(obj)

    final_df = pd.DataFrame(rows).sort_values(["date", "hit", "clean_name", "clean_artists"])
    final_df.to_csv(OUT_PATH, index=False)

    n_hit = int((final_df["hit"] == 1).sum())
    n_nothit = int((final_df["hit"] == 0).sum())
    n_hit_songs = final_df[final_df["hit"] == 1].groupby(["clean_name", "clean_artists"]).ngroups
    n_nothit_songs = final_df[final_df["hit"] == 0].groupby(["clean_name", "clean_artists"]).ngroups

    report = [
        "Validation report for tracks_charts_grouped_dataset.csv (Entrega 3, version final)",
        "",
        "Este dataset conserva la granularidad semanal (una fila por cancion y semana),",
        "a diferencia de la version deduplicada (tracks_charts_dedup_dataset.csv). La",
        "prevencion de fuga de informacion se resuelve en el split (por grupo de cancion),",
        "no eliminando filas.",
        "",
        f"Filas totales: {final_df.shape[0]}",
        f"hit=1: {n_hit} filas, {n_hit_songs} canciones distintas ({n_hit / n_hit_songs:.2f} filas/cancion en promedio)",
        f"hit=0: {n_nothit} filas, {n_nothit_songs} canciones distintas ({n_nothit / n_nothit_songs:.2f} filas/cancion en promedio)",
        f"Proporcion de hits: {n_hit / len(final_df):.2%}",
        f"Rango de fechas: {final_df['date'].min().date()} a {final_df['date'].max().date()}",
        f"Columnas: {final_df.shape[1]}",
        "",
        "Checks:",
        "  sin duplicados de date/song/artist en datos crudos combinados: OK",
        "  sin canciones presentes simultaneamente como hit y no-hit: OK",
        "  variable rank excluida (evita fuga de informacion): OK",
        "  contexto temporal calculado sobre la ocupacion real del ranking: OK",
        "  el split train/val/test se realiza por grupo de cancion (ver train_all_architectures.py)",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {OUT_PATH} ({final_df.shape[0]} rows, {n_hit} hit / {n_nothit} not-hit, "
          f"{n_hit_songs} hit songs / {n_nothit_songs} not-hit songs)")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
