import os
import re
from tabnanny import check
import numpy as np
import platform
import argparse
import subprocess
import pandas as pd
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
try:
    from ...utils.split import getDataJSON
except:
    import sys
    sys.path.insert(1,os.path.join(os.path.abspath('.'),"..",".."))
    from utils.split import getDataJSON


def svcBinaryClassifier():
    parser = argparse.ArgumentParser(description='SVC script AA@PAN')
    parser.add_argument('-t', type=str,
        help='Path to the jsonl-file with the text test dataset')
    parser.add_argument('-v', type=str,
        help='Path to the jsonl-file with ground test truth scores')
    parser.add_argument('-n', type=str,
        help='Path to the jsonl-file with the text train dataset')
    parser.add_argument('-y', type=str,
        help='Path to the jsonl-file with ground train truth scores')
    args = parser.parse_args()

    if not args.v:
        raise ValueError('The ground test truth path is required')
    if not args.t:
        raise ValueError('The test dataset is required')
    if not args.y:
        raise ValueError('The ground truth train path is required')
    if not args.n:
        raise ValueError('The train dataset is required')

    data = pd.DataFrame(getDataJSON(args.n)).set_index("id")
    data[['text1','text2']] = pd.DataFrame(data.pair.tolist(), index= data.index)

    del data["pair"]
    data2 = pd.DataFrame(getDataJSON(args.y)).set_index("id")
    data = pd.merge(data,data2,how='outer',left_index=True,right_index=True)
    del data2

    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
    X_2 = unigram_vectorizer.fit_transform(data["text1"])
    X_2 = X_2.toarray() - unigram_vectorizer.transform(data["text2"]).toarray()

    #Entrenamiento SVC
    clf = SVC(gamma='auto',probability=True)
    clf.fit(X_2, data["value"])
    
    data_test = pd.DataFrame(getDataJSON(args.t)).set_index("id")
    data_test[['text1','text2']] = pd.DataFrame(data_test.pair.tolist(), index= data_test.index)
    del data_test["pair"]
    data2 = pd.DataFrame(getDataJSON(args.v)).set_index("id")
    data_test = pd.merge(data_test,data2,how='outer',left_index=True,right_index=True)
    del data2

    #Predicción
    X_test = unigram_vectorizer.transform(data_test["text1"]).toarray() -\
            unigram_vectorizer.transform(data_test["text2"]).toarray()
    y_pred = clf.predict_proba(X_test)

    BASE_DIR = Path(__file__).resolve().parent
    predictions = []
    aux = []
    inicial = re.findall(r'([0-9]+)-[0-9]+-[0-9]+',data_test.index[0])[0]
    for x,y in zip(data_test.index,y_pred):
        actual = re.findall(r'([0-9]+)-[0-9]+-[0-9]+',x)[0]
        index = re.findall(r'[0-9]+-([0-9]+)-[0-9]+',x)[0]
        print(inicial,actual)
        if inicial == actual:
            aux.append({"index":index,"aux":x, "value":y[0]})
        else:
            predictions.append(aux)
            aux = []
            aux.append({"index":index,"aux":x, "value":y[0]})
            inicial = actual
    
    j=0
    for text in predictions:
        max_proba = text[0]["value"]
        max_index = text[0]["index"]
        for prediction in text:
            if max_proba < prediction["value"]:
                max_proba = prediction["value"]
                max_index = prediction["index"]
        len_index = int(prediction["index"])
        
        if len_index == 0:
            changes = [1]
        else:
            changes = []
            for i in range(len_index):
                if i==0 and int(max_index) == 0:
                    changes.append(1)
                elif i == int(max_index)-1:
                    changes.append(1)
                changes.append(0)
        with open(os.path.join(BASE_DIR,"prediction",f"prediction-problem-{re.findall(r'([0-9]+)-[0-9]+-[0-9]+',text[0]['aux'])[0]}.json"),"w+") as f:
            f.write(str({"changes":changes}).replace("'[","[").replace("']","]").replace("'",'"'))
        print("Predictions saved in:")
        print(os.path.join(BASE_DIR,"prediction"))                        
            
if __name__ == "__main__":
    svcBinaryClassifier()