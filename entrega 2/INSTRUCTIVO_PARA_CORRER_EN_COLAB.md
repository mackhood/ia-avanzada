# Pasos para correr el prototipo LSTM en Google Colab

## 1. Archivos necesarios

Antes de abrir Colab, tener a mano estos dos archivos:

```text
tracks_charts_classification_dataset.csv
modelo_lstm_hit_prediction_colab.py
```

Estan dentro de la carpeta:

```text
entrega_2_prototipo_lstm
```

## 2. Abrir Google Colab

Entrar a:

```text
https://colab.research.google.com/
```

Crear una notebook nueva.

## 3. Subir los archivos

En el panel izquierdo de Colab:

1. Tocar el icono de carpeta.
2. Tocar el boton de subir archivo.
3. Subir:

```text
tracks_charts_classification_dataset.csv
modelo_lstm_hit_prediction_colab.py
```

Alternativa: ejecutar esta celda y seleccionar los archivos:

```python
from google.colab import files
files.upload()
```

## 4. Ejecutar el modelo

En una celda nueva ejecutar:

```python
%run modelo_lstm_hit_prediction_colab.py
```

El script va a:

- cargar el dataset;
- armar la entrada temporal para la LSTM;
- dividir los datos en entrenamiento, validacion y prueba;
- entrenar el modelo;
- calcular metricas;
- generar predicciones;
- exportar el modelo y los preprocesadores.

## 5. Verificar que cargo bien

Al inicio deberia aparecer algo similar a:

```text
Dataset shape: (2019, 71)
Date range: 2021-05-22 to 2021-11-06
Train weeks: ...
Val weeks: ...
Test weeks: ...
LSTM input: (1506, 4, 9)
Song input: (1506, 39)
```

Eso confirma que el dataset se cargo correctamente.

## 6. Resultados esperados

Al final deberian aparecer metricas similares a:

```text
Accuracy: 0.9881
Precision: 0.9455
Recall: 1.0000
F1: 0.9720
ROC AUC: 0.9969
```

Tambien deberia aparecer:

```text
Saved artifacts:
modelo_lstm_hits.keras
context_scaler.joblib
song_preprocessor.joblib
best_threshold.joblib
modelo_lstm_metricas.txt
modelo_lstm_predicciones_test.csv
```

## 7. Descargar los archivos generados

En una celda nueva ejecutar:

```python
from google.colab import files

for f in [
    "modelo_lstm_hits.keras",
    "context_scaler.joblib",
    "song_preprocessor.joblib",
    "best_threshold.joblib",
    "modelo_lstm_metricas.txt",
    "modelo_lstm_predicciones_test.csv",
]:
    files.download(f)
```

## 8. Ver predicciones dentro de Colab

Para inspeccionar las predicciones:

```python
import pandas as pd

pred = pd.read_csv("modelo_lstm_predicciones_test.csv")
pred.head(20)
```

Para ver solo los errores:

```python
pred[pred["correct"] == False]
```

## 9. Recomendacion para compartir con el grupo

Compartir una carpeta con:

```text
notebook.ipynb
tracks_charts_classification_dataset.csv
modelo_lstm_hit_prediction_colab.py
pasos_colab_lstm.md
```

Si se comparte solo la notebook, los archivos subidos manualmente pueden no estar disponibles para otros usuarios.

