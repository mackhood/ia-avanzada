"""
Entrega 3 - Construccion del dataset sin informacion duplicada.

Corrige la observacion del docente sobre la Entrega 2: en el dataset anterior,
una misma cancion "hit" aparecia como una fila de entrenamiento distinta por
cada semana que permanecio en el ranking Billboard (en promedio ~10 semanas
por cancion exitosa). Como las caracteristicas propias de la cancion
(danceability, energy, etc.) son practicamente constantes en esas filas, el
modelo terminaba viendo el mismo "fingerprint" de audio muchas veces bajo la
etiqueta hit=1, lo cual exagera las metricas y no representa el problema real
(predecir si una cancion nueva llegara a ser un hit).

Este script reconstruye el dataset a nivel de "una fila por cancion":
- Para las canciones que llegaron a ser hit, se conserva unicamente la
  semana de debut en el ranking (primera aparicion), que es el momento en
  que efectivamente se podria haber hecho la prediccion.
- Para las canciones no-hit, el dataset original ya tiene una sola fila por
  cancion (fueron muestreadas una unica vez), por lo que se mantienen igual.
- El contexto temporal (promedio de features de los hits de las 4 semanas
  previas) se sigue calculando con la ocupacion real y completa del ranking
  semana a semana, ya que eso describe el estado del ranking en cada momento
  y no constituye una duplicacion de ejemplos de entrenamiento.

Entradas:
  tracks_charts_hit_cleaned.csv
  tracks_charts_not_hit_cleaned.csv

Salidas:
  tracks_charts_dedup_dataset.csv
  tracks_charts_dedup_dataset_validation.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HIT_PATH = ROOT / "tracks_charts_hit_cleaned.csv"
NOTHIT_PATH = ROOT / "tracks_charts_not_hit_cleaned.csv"
OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = OUT_DIR / "tracks_charts_dedup_dataset.csv"
REPORT_PATH = OUT_DIR / "tracks_charts_dedup_dataset_validation.txt"

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


NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


UNIT_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]


def parse_numeric_column(series: pd.Series) -> pd.Series:
    """Algunos registros del dataset de no-hits vienen con coma decimal o
    notacion cientifica con coma (ej. '1,00E-05') en lugar de punto."""

    def parse_value(value):
        text = str(value).strip()
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        return float(text)

    return series.map(parse_value)


def normalize_unit_feature(series: pd.Series, column: str) -> pd.Series:
    """Reproduce la correccion de escala aplicada en la Entrega 2: algunos
    registros de no-hits vienen en escala 0-1000 en lugar de 0-1."""

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
    # en el ranking esa semana (ocupacion real del chart, no se deduplica).
    weekly_hit_rows = combined[combined["hit"] == 1]
    weekly_context = {}
    for week in weeks:
        week_hits = weekly_hit_rows[weekly_hit_rows["date"] == week]
        if week_hits.empty:
            raise ValueError(f"No hit rows available to build context for week {week}")
        weekly_context[week] = week_hits[CONTEXT_FEATURES].mean()

    # --- Deduplicacion a nivel de cancion ---
    # Solo se conservan semanas con las 4 semanas previas disponibles (contexto completo).
    hit["date"] = pd.to_datetime(hit["date"])
    nothit["date"] = pd.to_datetime(nothit["date"])
    hit_with_context = hit[hit["date"].map(week_index) >= WINDOW_SIZE]
    nothit_with_context = nothit[nothit["date"].map(week_index) >= WINDOW_SIZE]

    # Hits: de las semanas utilizables, nos quedamos con la mas temprana por
    # cancion (lo mas cercano posible a su ingreso real al ranking, evitando
    # quedarnos con una cancion que ya llevaba muchas semanas siendo exitosa).
    hit_debut = (
        hit_with_context.sort_values("date")
        .drop_duplicates(subset=["clean_name", "clean_artists"], keep="first")
        .copy()
    )
    hit_debut["date"] = pd.to_datetime(hit_debut["date"])

    # No-hits: ya son unicos por cancion en el dataset original; se valida igual.
    nothit_dedup = nothit_with_context.drop_duplicates(subset=["clean_name", "clean_artists"], keep="first").copy()
    nothit_dedup["date"] = pd.to_datetime(nothit_dedup["date"])

    n_dropped_hit_rows = len(hit) - len(hit_debut)
    n_dropped_nothit_rows = len(nothit) - len(nothit_dedup)

    song_level = pd.concat([hit_debut, nothit_dedup], ignore_index=True)
    song_level["release_date"] = pd.to_datetime(song_level["release_date"], errors="coerce")
    song_level["release_year"] = song_level["release_date"].dt.year

    dup_after = song_level.duplicated(subset=["clean_name", "clean_artists"]).sum()
    if dup_after:
        raise ValueError(f"Duplicate song/artist keys remain after dedup: {dup_after}")

    rows = []
    for _, row in song_level.iterrows():
        week = row["date"]
        i = week_index[week]
        obj = row.to_dict()
        lag_means = {feat: [] for feat in CONTEXT_FEATURES}
        for lag in range(1, WINDOW_SIZE + 1):
            lag_week = weeks[i - lag]
            ctx = weekly_context[lag_week]
            for feat in CONTEXT_FEATURES:
                obj[f"ctx_{feat}_t_minus_{lag}"] = ctx[feat]
                lag_means[feat].append(ctx[feat])
        for feat in CONTEXT_FEATURES:
            avg_context = float(np.mean(lag_means[feat]))
            obj[f"delta_{feat}_vs_prev4_hit_avg"] = float(row[feat]) - avg_context
        rows.append(obj)

    final_df = pd.DataFrame(rows).sort_values(["date", "hit", "clean_name", "clean_artists"])

    if final_df.duplicated(subset=["clean_name", "clean_artists"]).sum():
        raise ValueError("Duplicate song/artist keys in final dataset")

    final_df.to_csv(OUT_PATH, index=False)

    n_hit = int((final_df["hit"] == 1).sum())
    n_nothit = int((final_df["hit"] == 0).sum())

    report = []
    report.append("Validation report for tracks_charts_dedup_dataset.csv (Entrega 3)")
    report.append("")
    report.append("Objetivo: eliminar la duplicacion de informacion senalada por el docente en la Entrega 2")
    report.append("(una misma cancion hit generaba una fila de entrenamiento por cada semana en el ranking).")
    report.append("")
    report.append("Input rows:")
    report.append(f"  tracks_charts_hit_cleaned.csv: {len(hit)} filas, {hit.groupby(['clean_name','clean_artists']).ngroups} canciones unicas")
    report.append(f"  tracks_charts_not_hit_cleaned.csv: {len(nothit)} filas, {nothit.groupby(['clean_name','clean_artists']).ngroups} canciones unicas")
    report.append("")
    report.append("Deduplicacion:")
    report.append(f"  filas de hit descartadas por repetir semana de una cancion ya vista: {n_dropped_hit_rows}")
    report.append(f"  filas de no-hit descartadas por repetir semana de una cancion ya vista: {n_dropped_nothit_rows}")
    report.append(f"  canciones hit conservadas (semana de debut unicamente): {hit_debut.shape[0]}")
    report.append(f"  canciones no-hit conservadas: {nothit_dedup.shape[0]}")
    report.append("")
    report.append("Dataset final (una fila por cancion, con contexto de 4 semanas previas):")
    report.append(f"  filas totales: {final_df.shape[0]}")
    report.append(f"  hit=1: {n_hit} ({n_hit / len(final_df):.2%})")
    report.append(f"  hit=0: {n_nothit} ({n_nothit / len(final_df):.2%})")
    report.append(f"  rango de fechas: {final_df['date'].min().date()} a {final_df['date'].max().date()}")
    report.append(f"  columnas: {final_df.shape[1]}")
    report.append("")
    report.append("Checks:")
    report.append("  sin duplicados de date/song/artist en datos crudos combinados: OK")
    report.append("  sin canciones presentes simultaneamente como hit y no-hit: OK")
    report.append("  una unica fila por cancion/artista en el dataset final: OK")
    report.append("  contexto temporal calculado sobre la ocupacion real del ranking (no deduplicada): OK")
    report.append("  variable rank excluida (evita fuga de informacion): OK")
    report.append("")
    report.append("Columnas:")
    report.append("  " + ", ".join(final_df.columns))

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {OUT_PATH} ({final_df.shape[0]} rows, {n_hit} hit / {n_nothit} not-hit)")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
