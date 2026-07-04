# Entrega 3 - Evolucion del Prototipo del Sistema Inteligente

**Grupo B** | Materia: Inteligencia Artificial Avanzada

## Que cambio respecto de la Entrega 2

El docente senalo dos observaciones sobre la [Entrega 2](../entrega%202/README.md): (1) evitar informacion duplicada en el dataset, y (2) evaluar otras arquitecturas para la rama temporal (GRU y/o ConvNet). Ambas se abordaron en esta entrega.

### Correccion metodologica

El problema real de la Entrega 2 no era que una misma cancion generara varias filas (una por cada semana en el ranking) -eso es parte real del fenomeno, ya que cada semana tiene un contexto de tendencias distinto-, sino que el split train/val/test era **cronologico a nivel de fila**: una misma cancion podia tener semanas en entrenamiento y semanas en prueba, filtrando su "fingerprint" de audio (que si es constante) entre ambos conjuntos.

Se probaron dos correcciones que fueron descartadas antes de llegar a la solucion final:

1. **Deduplicar a una fila por cancion.** Se descarto porque tiraba la senal real de persistencia en el ranking y reducia la clase positiva de 769 a 79 hits, dejando un conjunto de prueba con solo 20 casos positivos. Conservado como registro en `tracks_charts_dedup_dataset.csv`, `scripts/build_dedup_dataset.py` y `scripts/train_all_architectures.py`.
2. **Split aleatorio estratificado por fila sobre el dataset completo (sin agrupar por cancion).** Se descarto porque 56 canciones quedaban repartidas entre entrenamiento y prueba (43 entre train/val, 41 entre val/test), reproduciendo la misma fuga de informacion de la Entrega 2 sobre un dataset distinto, con metricas aun mas altas (F1 99,22%) precisamente por esa fuga.

**Solucion final:** dataset completo (2019 filas, 769 hits, igual que la Entrega 2) con split por **grupo de cancion**: se implementa un algoritmo de asignacion voraz (greedy bin-packing) que asigna cada cancion completa (con todas sus semanas) a un unico subconjunto, buscando aproximar las proporciones 60% train / 15% validacion / 25% test. Resultado: 60,0% / 15,0% / 25,0% exacto, y se verifica de forma explicita que ninguna cancion aparece en mas de un subconjunto.

### Comparacion de arquitecturas

Se generalizo la rama temporal del modelo hibrido para poder intercambiar la capa utilizada, manteniendo igual el resto de la arquitectura (rama de la cancion, concatenacion, cabeza de clasificacion):

- **LSTM** (linea base de la Entrega 2): capa LSTM de 32 unidades.
- **GRU**: capa GRU de 32 unidades, menos parametros que LSTM.
- **Conv1D (ConvNet)**: dos capas Conv1D (32 filtros, kernel 2) + GlobalMaxPooling1D, sin estado recurrente.

## Resultados (test, split por grupo de cancion, 504 filas, 192 hits)

| Arquitectura | Umbral | Accuracy | Precision | Recall | F1-score | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|---|
| **LSTM (recomendada)** | 0,39 | **98,02%** | 95,50% | **99,48%** | **97,45%** | 99,80% | 99,68% |
| GRU | 0,84 | 96,23% | 98,87% | 91,15% | 94,85% | **99,86%** | 99,76% |
| Conv1D | 0,13 | 95,04% | 88,48% | 100,00% | 93,89% | 99,70% | 99,45% |

### Comparacion con la Entrega 2

| Version | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---|---|---|---|---|
| Entrega 2 (LSTM, split cronologico con fuga) | 98,81% | 94,55% | 100,00% | 97,20% | 99,69% |
| Variante descartada (split aleatorio por fila, con fuga) | 99,41% | 98,46% | 100,00% | 99,22% | 99,63% |
| **Entrega 3 final (LSTM, split por grupo, sin fuga)** | 98,02% | 95,50% | 99,48% | **97,45%** | **99,80%** |

El hallazgo clave: al eliminar por completo el riesgo de fuga de informacion, las metricas de LSTM se mantuvieron practicamente iguales a las de la Entrega 2, e incluso mejoraron levemente en F1 y ROC AUC. Esto indica que el buen desempeno de la Entrega 2 no era, en su mayor parte, un artefacto de la fuga de informacion, sino que el modelo efectivamente aprende patrones generalizables.

## Archivos de esta entrega

| Archivo / carpeta | Descripcion |
|---|---|
| `Entrega N° 3_ Evolucion del Sistema Inteligente.docx` | Informe final de la entrega |
| `explicacion_entrega3.html` | Resumen visual con metricas, figuras y comparaciones |
| `tracks_charts_grouped_dataset.csv` | Dataset final: cancion x semana completo (2019 filas), sin deduplicar |
| `split_asignacion_canciones.csv` | Subconjunto (train/val/test) asignado a cada cancion |
| `scripts/build_grouped_dataset.py` | Construye el dataset completo cancion x semana |
| `scripts/train_all_architectures_grouped.py` | Split por grupo de cancion + entrenamiento/evaluacion de LSTM, GRU y Conv1D |
| `scripts/plot_architecture.py` | Genera el diagrama conceptual de la arquitectura |
| `scripts/plot_comparison.py` | Genera el grafico comparativo Entrega 2 vs Entrega 3 |
| `scripts/build_report.py` | Genera el informe .docx a partir de la plantilla de la catedra |
| `resultados_lstm_grouped/`, `resultados_gru_grouped/`, `resultados_conv1d_grouped/` | Modelo (.keras), preprocesadores, metricas, predicciones y figuras por arquitectura (version final) |
| `comparacion_arquitecturas_grouped.csv` / `.md` | Tabla comparativa de las 3 arquitecturas (version final) |
| `tracks_charts_dedup_dataset.csv`, `resultados_lstm/`, `resultados_gru/`, `resultados_conv1d/`, `scripts/build_dedup_dataset.py`, `scripts/train_all_architectures.py` | Primera iteracion (deduplicacion total), conservada como registro del proceso iterativo; reemplazada por la version final |

## Como reproducir

```bash
pip install tensorflow-cpu pandas scikit-learn joblib matplotlib python-docx
python scripts/build_grouped_dataset.py
python scripts/train_all_architectures_grouped.py
python scripts/plot_comparison.py
python scripts/build_report.py
```

## Lecciones aprendidas

- No toda repeticion de datos es "informacion duplicada" en sentido negativo: el problema no era la repeticion en si, sino permitir que cruzara la frontera train/test.
- La primera solucion no siempre es la correcta: deduplicar por completo parecia la forma mas directa de resolverlo, pero tiraba informacion valiosa del problema real.
- El lugar correcto para resolver una fuga de informacion suele ser el split, no el dataset.
- Con grupos de tamano desigual, un split aleatorio simple no alcanza: hizo falta un algoritmo de asignacion voraz (greedy bin-packing).
- Mejores metricas no siempre significan un mejor sistema: lo importante es que la metrica sea confiable.
- "Usar el dataset completo" no alcanza si el split sigue siendo por fila: se confirmo con un segundo caso (56 canciones filtradas) que evitar la duplicacion y evitar la fuga de informacion son dos cosas distintas.

## Stack tecnologico

- Python 3 (entorno local, TensorFlow-CPU 2.21)
- TensorFlow / Keras
- scikit-learn
- pandas / numpy
- python-docx (generacion del informe)
