***
### Descripción

Los datos proporcionados por el PAN22 para esta tarea ya se encuentran separados en conjuntos de: `train`, `validation`, `test`, los primeros dos cuentan con etiquetas y el úlitmo no, la carpeta `dataset1` corresponde a los datos para la primer tarea que el PAN propone para este reto.

Los textos se encuentran en formato `.txt`, tienen el mismo nombre y diferente enumeración `(problem-*.txt)` , mientras que las etiquetas se encuentran en archivos `.json` siguiendo el mismo formato `(truth-problem-*.json)`.

Para usar los baselines es necesario preparar nuestros datos para que sean adecuados para los modelos, en este caso se utilizaron los baselines para la tarea de **Autorship-Verification** haciendo algunas modificaciones, por lo que para tener los datos de forma correcta es necesario correr el script `prepare_data.py` de la siguiente forma:

***
### Datos de entrenamiento
```sh
python3 prepare_data.py -r "./pan22/dataset1/train" -o "./pan22/dataset1/prepared/train" -y 1
```
Este instrucción creará dos archivos llamados `train_truth.jsonl` y `train.jsonl`, los cuales se ven de la siguiente manera:
> __train_truth.jsonl__
```
{"id": "1-1-2", "same": 1}
{"id": "1-1-3", "same": 1}
{"id": "1-1-4", "same": 0}
{"id": "1-2-3", "same": 1}
{"id": "1-2-4", "same": 1}
{"id": "1-3-4", "same": 0}
...
``` 
Para este caso el `id` se conforma de la siguiente manera <NUMERO_DOCUMENTO>-<LINEA DEL RENGLÓN>-<LINEA DEL OTRO RENGLÓN CON EL QUE SE COMPARO>, mientas que `same` indica si los dos renglones corresponden al mismo autor o no.

> __train.jsonl__
```
{"id": "1-1-2", "pair": ["I use squid on RHEL6 and I want that authentication is via AD windows 2008, I already joined the server to the windows domain and all users is already seen by wbinfo -u wbinfo -g but wbmin -t show error below : \n", "checking the trust secret for domain TELMA via RPC calls failed\n"]}
{"id": "1-1-3", "pair": ["I use squid on RHEL6 and I want that authentication is via AD windows 2008, I already joined the server to the windows domain and all users is already seen by wbinfo -u wbinfo -g but wbmin -t show error below : \n", "I followed this tuto https://www.dalemacartney.com/2012/0...nd-simple-way/ and all is fine and normally all user on domain doesn't require authentication but when I configured the browser to point to the proxy it's always requiring authentication and showing error below on /var/log/squid/cache.log : \n"]}
{"id": "1-1-4", "pair": ["I use squid on RHEL6 and I want that authentication is via AD windows 2008, I already joined the server to the windows domain and all users is already seen by wbinfo -u wbinfo -g but wbmin -t show error below : \n", "2014/07/31 15:47:07| squid_kerb_auth: ERROR: gss_acquire_cred() failed: Unspecified GSS failure. Minor code may provide more information. Unknown error\n"]}
{"id": "1-2-3", "pair": ["checking the trust secret for domain TELMA via RPC calls failed\n", "I followed this tuto https://www.dalemacartney.com/2012/0...nd-simple-way/ and all is fine and normally all user on domain doesn't require authentication but when I configured the browser to point to the proxy it's always requiring authentication and showing error below on /var/log/squid/cache.log : \n"]}
{"id": "1-2-4", "pair": ["checking the trust secret for domain TELMA via RPC calls failed\n", "2014/07/31 15:47:07| squid_kerb_auth: ERROR: gss_acquire_cred() failed: Unspecified GSS failure. Minor code may provide more information. Unknown error\n"]}
{"id": "1-3-4", "pair": ["I followed this tuto https://www.dalemacartney.com/2012/0...nd-simple-way/ and all is fine and normally all user on domain doesn't require authentication but when I configured the browser to point to the proxy it's always requiring authentication and showing error below on /var/log/squid/cache.log : \n", "2014/07/31 15:47:07| squid_kerb_auth: ERROR: gss_acquire_cred() failed: Unspecified GSS failure. Minor code may provide more information. Unknown error\n"]}
...
``` 

Nuevamente el `id` se conforma de la manera que se explico previamente, mientas que `pair` contiene los pares de textos.


***
### Datos de validación/prueba
```sh
python3 prepare_data.py -r "./pan22/dataset1/validation" -o "./pan22/dataset1/prepared/validation" -y 0
```
Este instrucción creará un archivo llamado `validation.json` que se ve de la siguiente manera:
> __validation.json__
```
[
    [
        {
            "id": "1-1-2",
            "pair": [
                "If you can handle a slight delay in the data you are reporting on then a easy solution would be to restore the database less frequently.\n",
                "You can still backup and copy every 5 minutes but just change the frequency of the restore job to once an hour or whatever is appropriate. \n"
            ]
        },
        {
            "id": "1-2-3",
            "pair": [
                "You can still backup and copy every 5 minutes but just change the frequency of the restore job to once an hour or whatever is appropriate. \n",
                "Users may still get disconnected but the frequency will be lower.\n"
            ]
        },
        {
            "id": "1-3-4",
            "pair": [
                "Users may still get disconnected but the frequency will be lower.\n",
                "If you need to keep online access to the data, you've got a couple of options (assuming you're wanting to use just SQL native functionality for the solution, if you are open to 3rd party software and/or hardware, you've got quite a few other options):\n"
            ]
        },
        {
            "id": "1-4-5",
            "pair": [
                "If you need to keep online access to the data, you've got a couple of options (assuming you're wanting to use just SQL native functionality for the solution, if you are open to 3rd party software and/or hardware, you've got quite a few other options):\n",
                "1) Replication - most likely transactional replication and a single read-only subscriber (msdn is a good start for an overview, I'd post a link but I can only use 1 at the moment, just google \"sql server replication msdn\" and it will be at the top)\n"
            ]
        },
        {
            "id": "1-5-6",
            "pair": [
                "1) Replication - most likely transactional replication and a single read-only subscriber (msdn is a good start for an overview, I'd post a link but I can only use 1 at the moment, just google \"sql server replication msdn\" and it will be at the top)\n",
                "2) Keep your log shipping configuration to get data to the secondary server and leverage database snapshots combined with a common database and rotating synonyms (see here for details on this type of architecture). This will only work if you are using the enterprise edition on the secondary server (only edition that supports snapshots)."
            ]
        }
    ],
    ...
]
``` 
Para este caso el `id` se conforma de la siguiente manera <NUMERO_DOCUMENTO>-<LINEA DEL RENGLÓN>-<LINEA DEL RENGLÓN SUBSECUENTE>, mientas que `pair` contiene los pares de textos, para este caso no nos interesa tener todas las combianaciones posibles entre parrafos, sólo queremos en que par de textos ocurren la probabilidad más pequeña de que compartan el mismo autor para identificar en los modelos donde ocurrió el cambio, para calcular la métrica que solicita el PAN es necesario el folder con todos los documentos `.json` que contienen las etiquetas.

***

python3 prepare_data.py -r "./pan22/dataset1/validation" -o "./pan22/dataset1/prepared/validation" -y 0

### Argumentos para el script `prepare_data.py`
| Args   | Description                                                                  |
|--------|------------------------------------------------------------------------------|
| `-r`   | Folder con los datos .txt y .json                                            |
| `-o`   | Folder donde se almacenarán los arhivo(s) generados                          |
| `-y`   | 0 si se tratan de datos de validación/prueba o 1 para datos de entrenamiento |
