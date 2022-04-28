# StyleChangeDetection2022

***
## Baseline 

Este repositorio contiene las implementaciones de los baselines propuestos para la tarea de verificación de autoría del PAN2021 adecuados para que resuelvan el problema de la detección de cambio de estilo para el PAN2022.

- **Support Vector**

Este modelo se basa en conjunto de ejemplos de entrenamiento (de muestras) podemos etiquetar las clases y entrenar una [SVM](https://en.wikipedia.org/wiki/Support-vector_machine) para construir un modelo que prediga la clase de una nueva muestra. Intuitivamente, una SVM es un modelo que representa a los puntos de muestra en el espacio, separando las clases a 2 espacios lo más amplios posibles mediante un hiperplano de separación definido como el vector entre los 2 puntos, de las 2 clases, más cercanos al que se llama vector soporte.

- **Cosine Similarity**

En esta carpeta se encuentra una solución rápida a la tarea PAN2020 sobre verificación de autoría. Todos los documentos se representan usando un modelo de Bag of character ngrams, eso es TFIDF ponderado. La [semejanza del coseno](https://en.wikipedia.org/wiki/Cosine_similarity) entre cada par de documentos en el conjunto de datos de calibración es calculado. Finalmente, las similitudes resultantes son optimizadas y proyectadas a través de un simple reescalado, para que puedan funcionar como pseudo-probabilidades, que indican la probabilidad de que un par de documentos es un par del mismo autor.

- **Compresión de Textos**

El método basado en compresión de textos (compression method calculating cross-entropy), se propone en [Teahan & Harper](https://link.springer.com/chapter/10.1007/978-94-017-0171-6_7). La implementación contenida en este repositorio es una ligera modificación: en lugar de proponer una regla de decisión basada en un umbral, se usa una regresión logística para hacer la clasificación.

***
## Prerequisitos
  - Using venv
  ```sh
  # Create the virtual environment
  python3.9 -m venv env
  # Activate the virtual environment
  source env/bin/activate
  # Install the dependencies
  pipenv install -r requirements.txt
  ```

***
## corpus

Contiene los datos para resolver la tarea de cambio de estilo del PAN2022, también es necesario correr un script para preparar los datos para los modelos, por lo que antes de ejecutar cualquier modelo es necesario revisar este carpeta.

***
## utils

La carpeta utils contiene scripts que contienen funciones de uso común en los diferentes modelos