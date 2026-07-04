# IA Avanzada - Grupo B

**Materia:** Inteligencia Artificial Avanzada

Prototipo de red neuronal hibrida para la prediccion de hits musicales. El modelo clasifica canciones como "hit" o "no hit" combinando sus atributos sonoros (Spotify) con el contexto de tendencias musicales de las semanas previas (Billboard Hot 100).

Cada entrega del trabajo practico tiene su propia carpeta con informe, codigo, dataset y resultados. Ver el README de cada una para el detalle completo.

## [Entrega 2](entrega%202/README.md) - Implementacion del Prototipo

Primera version del Sistema Inteligente: dataset a nivel de cancion x semana (2019 filas), red hibrida LSTM + rama densa de atributos de la cancion, split cronologico. Resultado en test: F1 97,20%, ROC AUC 99,69%.

## [Entrega 3](entrega%203/README.md) - Evolucion del Prototipo

Corrige una fuga de informacion detectada en el split de la Entrega 2 (una misma cancion podia tener semanas en train y en test) mediante un split por grupo de cancion, y compara tres arquitecturas para la rama temporal (LSTM, GRU, Conv1D). La version final (LSTM, split por grupo) da F1 97,45% y ROC AUC 99,80% sobre un conjunto de prueba mas grande y sin riesgo de fuga, confirmando que el buen desempeno de la Entrega 2 era en gran medida genuino.

## Stack tecnologico

- Python 3
- TensorFlow / Keras
- scikit-learn
- pandas / numpy
- python-docx (generacion de informes)
