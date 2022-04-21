# Compression method calculating cross-entropy

## Prerequisites

***
  - Python 3.6.x


## Run
***
> __A continuación asumimos que el directorio de trabajo es la raíz del repositorio.__  

Para ejecutar el script, usando los datos base de pan14:
  ```sh
  python3 text_compression.py -i "../../corpus/pan14/test.jsonl" \
  -v "../../corpus/pan14/test_truth.jsonl" \
  -m "./model/model_pan14.joblib"
  ```
Output:
  ```sh
  Predictions saved in:
  ~/AuthorshipVerification-Baseline/baseline_methods/TextCompression/prediction/pan14_pred.jsonl
  ```
Para ejecutar el script, usando los datos base de pan15:
  ```sh
  python3 text_compression.py -i "../../corpus/pan15/test.jsonl" \
  -v "../../corpus/pan15/test_truth.jsonl" \
  -m "./model/model_pan15.joblib"
  ```
Output:
  ```sh
  Predictions saved in:
  ~/AuthorshipVerification-Baseline/baseline_methods/TextCompression/prediction/pan15_pred.jsonl
  ```

| Args   | Description                                    |
|--------|------------------------------------------------|
| `-i`   | test.jsonl file with relative route            |
| `-v`   | test_truth.jsonl file with relative route      |
| `-m`   | ruta del modelo (de regresión) entrenado a usar para la clasificación de los textos a partir de sus entropías cruzadas |
