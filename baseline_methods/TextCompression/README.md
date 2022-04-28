# Compression method calculating cross-entropy


***  
## Ejecutar
  
Para ejecutar el script, usando los datos del pan22 ya preparados (revisar carpeta `corpus` en la raíz del proyecto) y sin un modelo previamente entrenado:

```sh
python3 main.py -train_pairs="../../corpus/pan22/dataset1/prepared/train/train.jsonl" \ 
    -train_truth="../../corpus/pan22/dataset1/prepared/train/train_truth.jsonl" \
    -test_pairs="../../corpus/pan22/dataset1/prepared/validation/validation.json" \
    -test_truth="../../corpus/pan22/dataset1/validation"
```

En caso de que ya cuente con un modelo entrenado `.joblib` los argumentos cambian a:

```sh
python3 main.py -model="./model/model_pan.joblib" \
    -test_pairs="../../corpus/pan22/dataset1/prepared/validation/validation.json" \
    -test_truth="../../corpus/pan22/dataset1/validation"
```

Salida:
El programa guarda dentro de la carpeta `prediction` todas las prediciones con el nombre `prediction-problem-*.json` donde `*` corresponde al número del documento predicho y además cálcula el valor de la métrica solicitada por el PAN guardadolo en la carpeta `resultados/TestCompression` que se encuentra en la raíz del proyecto, para el caso de que no cuente con un modelo y allá ejecutado un comando similar al primer ejemplo el script guardará el modelo entrenado  en `model/model_pan.joblib`.

### Argumentos de `main.py`

| Args   		  | Description                                                                                                      |
|-----------------|------------------------------------------------------------------------------------------------------------------|
| `-train_pairs`  | Archivo .jsonl procesado para el entrenamiento (opcional si cuenta con un modelo .joblib)                        |
| `-train_truth`  | Archivo .jsonl procesado que contiene las etiquetas del entrenamiento (opcional si cuenta con un modelo .joblib) |
| `-test_pairs`   | Archivo .json procesado que contiene los pares de textos para validación                                         |
| `-test_truth`   | Ruta donde se encuentran las etiquetas para el conjunto de validación                                            |
| `-model`        | Modelo entrenado previamente (si no cuenta con uno ejecutar con los argumentos -train_pairs y -train_truth       |