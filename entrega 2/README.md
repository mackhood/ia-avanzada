# Entrega 2 - Prototipo del Sistema Inteligente

**Grupo B** | Materia: Inteligencia Artificial Avanzada

## Descripcion

Prototipo de red neuronal LSTM para la prediccion de hits musicales. El modelo clasifica canciones como "hit" o "no hit" en base a su historial de apariciones en charts semanales y sus atributos sonoros.

## Problema

Dado un dataset de canciones con sus posiciones semanales en charts (mayo–noviembre 2021), predecir si una cancion va a ingresar o mantenerse en el top de listas en la semana siguiente.

## Dataset

- **Archivo:** `tracks_charts_classification_dataset.csv`
- **Filas:** 2019 | **Columnas:** 71
- **Rango:** 2021-05-22 a 2021-11-06
- **Split temporal:**
  - Train: 2021-05-22 a 2021-09-11 (1506 filas)
  - Validacion: 2021-09-18 a 2021-10-09 (261 filas)
  - Test: 2021-10-16 a 2021-11-06 (252 filas)

## Arquitectura del modelo

El modelo combina dos ramas de entrada:

- **Rama contextual (LSTM):** secuencia de 4 semanas de contexto con 9 features temporales cada una
- **Rama de atributos de cancion:** 39 features estaticas del track (genero, energia, danceability, etc.)

Ambas ramas se concatenan y pasan por capas densas con Dropout antes de la salida binaria.

## Resultados

| Conjunto    | Accuracy | Precision | Recall | F1     | ROC AUC |
|-------------|----------|-----------|--------|--------|---------|
| Validacion  | 0.9655   | 0.8714    | 1.0000 | 0.9313 | 1.0000  |
| **Test**    | **0.9881** | **0.9455** | **1.0000** | **0.9720** | **0.9969** |

Umbral optimo seleccionado por F1 en validacion: **0.82**

> **Nota (ver Entrega 3):** este split fue posteriormente identificado como cronologico a nivel de fila, lo que permitia que una misma cancion tuviera semanas en train y en test. La [Entrega 3](../entrega%203/README.md) corrige este problema con un split por grupo de cancion.

## Archivos del repositorio

| Archivo | Descripcion |
|---|---|
| `modelo_lstm_hit_prediction_colab.py` | Script principal del modelo (entrenar y exportar) |
| `modelo_lstm_hits.keras` | Modelo entrenado exportado |
| `context_scaler.joblib` | Scaler para las features contextuales |
| `song_preprocessor.joblib` | Preprocesador de atributos de cancion |
| `best_threshold.joblib` | Umbral optimo de clasificacion |
| `modelo_lstm_metricas.txt` | Metricas completas del entrenamiento |
| `modelo_lstm_predicciones_test.csv` | Predicciones sobre el set de test |
| `tracks_charts_classification_dataset.csv` | Dataset completo |
| `tracks_charts_classification_dataset_validation.txt` | Descripcion del dataset de validacion |
| `INSTRUCTIVO_PARA_CORRER_EN_COLAB.md` | Guia paso a paso para reproducir en Google Colab |

## Como reproducir

Ver el archivo [INSTRUCTIVO_PARA_CORRER_EN_COLAB.md](INSTRUCTIVO_PARA_CORRER_EN_COLAB.md) para los pasos detallados.

En resumen:

1. Abrir [Google Colab](https://colab.research.google.com/)
2. Subir `tracks_charts_classification_dataset.csv` y `modelo_lstm_hit_prediction_colab.py`
3. Ejecutar `%run modelo_lstm_hit_prediction_colab.py`
4. El script entrena el modelo, calcula metricas y exporta todos los artefactos

## Stack tecnologico

- Python 3
- TensorFlow / Keras
- scikit-learn
- pandas / numpy
