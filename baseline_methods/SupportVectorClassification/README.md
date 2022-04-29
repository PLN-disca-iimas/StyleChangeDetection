# Authorship Verification Using SVC

***  
## Ejecutar
  
Para ejecutar el script, usando los datos del pan22 ya preparados (revisar carpeta `corpus` en la raíz del proyecto):

```sh
python3 main.py -train_pairs="../../corpus/pan22/dataset1/prepared/train/train.jsonl" \
    -train_truth="../../corpus/pan22/dataset1/prepared/train/train_truth.jsonl" \ 
    -test_pairs="../../corpus/pan22/dataset1/prepared/validation/validation.json" \
    -test_truth="../../corpus/pan22/dataset1/validation"
```

Salida:
El programa guarda dentro de la carpeta `prediction` todas las prediciones con el nombre `prediction-problem-*.json` donde `*` corresponde al número del documento predicho y además cálcula el valor de la métrica solicitada por el PAN guardadolo en la carpeta `resultados/SupportVectorClassification` que se encuentra en la raíz del proyecto.


### Argumentos de `main.py`

| Args   		  | Description                                                              |
|-----------------|--------------------------------------------------------------------------|
| `-input_pairs`  | Archivo .jsonl procesado para el entrenamiento                           |
| `-input_truth`  | Archivo .jsonl procesado que contiene las etiquetas del entrenamiento    |
| `-test_pairs`   | Archivo .json procesado que contiene los pares de textos para validación |
| `-test_truth`   | Ruta donde se encuentran las etiquetas para el conjunto de validación    |