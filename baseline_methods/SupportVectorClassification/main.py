import os
import platform
import re
import numpy as np
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
    parser.add_argument('-train_pairs', type=str, required=True,
                        help='Path to the jsonl-file with the train pairs')
    parser.add_argument('-train_truth', type=str, required=True,
                        help='Path to the ground truth jsonl-file for the train pairs')
    parser.add_argument('-test_pairs', type=str, required=True,
                        help='Path to the json-file with the validation/test pairs')
    parser.add_argument('-test_truth', type=str, required=True,
                        help='Path to the ground truth-file for the validation folder')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent

    data = pd.DataFrame(getDataJSON(args.train_pairs)).set_index("id")
    data[['text1','text2']] = pd.DataFrame(data.pair.tolist(), index= data.index)

    del data["pair"]
    data2 = pd.DataFrame(getDataJSON(args.train_truth)).set_index("id")
    data = pd.merge(data,data2,how='outer',left_index=True,right_index=True)
    del data2

    unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w+\b', min_df=1)
    X_2 = unigram_vectorizer.fit_transform(data["text1"])
    X_2 = X_2.toarray() - unigram_vectorizer.transform(data["text2"]).toarray()
    Y = data["value"]

    #Entrenamiento SVC
    #----------------------------------kernel lineal
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_2, Y)    

    #Conjunto de test/validation
    with open(args.test_pairs) as f:
        data = json.load(f)
    for dict in data:
        pred = []
        conf = []
        for dict_line in dict:
            X_test = unigram_vectorizer.transform([dict_line["pair"][0]]).toarray() -\
                     unigram_vectorizer.transform([dict_line["pair"][1]]).toarray()
            pred.append(clf.predict_proba(X_test)[0][1])
            conf.append(0)

        conf[np.argmin(pred)] = 1
        problem_number = int(re.findall(r'([0-9]+)-[0-9]+-[0-9]+',dict_line["id"])[0])
        with open(os.path.join(BASE_DIR,"prediction",f"prediction-problem-{problem_number}.json"),"w+") as f:
            json.dump({"changes":conf},f)
    
    PREDICTION_DIR = os.path.join(BASE_DIR,".","prediction") 
    OUTPUT_DIR = os.path.join(BASE_DIR,"..","..","resultados","CosineSimilarity")

    if "Windows" in platform.system():
        subprocess.run(["python","../../utils/evaluator_2022.py","-p",
            PREDICTION_DIR,"-t",args.test_truth,"-o",
            OUTPUT_DIR], capture_output=True)
    else:
        subprocess.run(["python3","../../utils/evaluator_2022.py","-p",
            PREDICTION_DIR,"-t",args.test_truth,"-o",
            OUTPUT_DIR], capture_output=True)
            
if __name__ == "__main__":
    svcBinaryClassifier()