import gensim
import os
import re
import numpy as np
import json as js
import argparse
import pandas as pd
import sklearn
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from gensim.models import KeyedVectors
from gensim import models



def load_model(model_path):
  modelW2V = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

  return modelW2V


def getDataJSON(route):
    with open(route,"r",encoding="utf-8") as f:
        result = [js.loads(jline.replace("diff","value")) for jline in f.read().splitlines()]
    return result


def word2vec(datadir,modelw2v):
  aux = os.path.join(datadir, "train.jsonl")
  data = pd.DataFrame(getDataJSON(aux)).set_index("id")
  data[['text1','text2']] = pd.DataFrame(data.pair.tolist(), index= data.index)

  aux = os.path.join(datadir, "train_truth.jsonl")

  del data["pair"]
  data2 = pd.DataFrame(getDataJSON(aux)).set_index("id")
  data = pd.merge(data,data2,how='outer',left_index=True,right_index=True)
  del data2

  X_0 = data["text1"].apply(str.lower)
  X_0 = X_0.apply(str.split)
  X_0 = X_0.apply(modelw2v.get_mean_vector, pre_normalize=False, post_normalize=True)
  X_1 = data["text2"].apply(str.lower)
  X_1 = X_1.apply(str.split)
  X_1 = X_1.apply(modelw2v.get_mean_vector, pre_normalize=False, post_normalize=True)
  X = X_0 - X_1 
  X = X.apply(np.absolute)
  Y = data["value"]
  return X,Y

def SVM(X,Y,exitmodel):
  print('%%%%%%TRAINING MODEL%%%%%%')
  X_train=np.array(X.tolist())
  Y_train=np.array(Y.tolist())
  svm_model = SVC(kernel='linear')
  svm_model.fit(X_train, Y_train)
  joblib.dump(svm_model,exitmodel)

def reshape_array(arr):
    return arr.reshape(1, -1)

def redimencionarPCA(X):
  pca = PCA(n_components=2)
  X_vectores = X.apply(pd.Series)
  pca.fit(X_vectores.T)
  X_PCAvec = pd.DataFrame(pca.components_.T,index=X_vectores.index)
  return X_PCAvec

def redimencionarTSNE(X):
  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
  X_vectores = X.apply(pd.Series)
  X_vec = tsne.fit_transform(X_vectores)
  X_TSNEvec = pd.DataFrame(X_vec,index=X_vectores.index)
  print(X_TSNEvec)
  return X_TSNEvec

def redimensionarISOMAP(X):
  isomap=Isomap(n_components=2)
  X_vectores = X.apply(pd.Series)
  X_vec = isomap.fit_transform(X_vectores)
  X_ISOMAPvec = pd.DataFrame(X_vec,index=X_vectores.index)
  print(X_ISOMAPvec)
  return X_ISOMAPvec
   

def plot_points_with_labels(vectors_df, labels_df):
  # Asegúrate de que las etiquetas y los vectores tengan el mismo índice
  labels_df.to_frame()
  # Separa los vectores según las etiquetas
  vectors_label_0 = vectors_df[labels_df == 0]
  vectors_label_1 = vectors_df[labels_df == 1]
    
  # Crea un gráfico de puntos
  plt.figure(figsize=(8, 6))
  plt.scatter(vectors_label_1.iloc[:, 0], vectors_label_1.iloc[:, 1], color='red', label='Con cambio')
  plt.scatter(vectors_label_0.iloc[:, 0], vectors_label_0.iloc[:, 1], color='blue', label='Sin cambio')
    
    
  # Configura las etiquetas y el título
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Embeddings Isomap')
   
  # Muestra la leyenda
  plt.legend()
   
  # Muestra el gráfico
  plt.show()


def main():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-m', type=str, help='Directorio del modelo de Word2Vec formato .bin')
  parser.add_argument('-d', type=str, help='Directorio del dataset')
  parser.add_argument('-p', type=str, help='Directorio de la salida del modelo SVM')
  args = parser.parse_args()

  if not args.m:
    print('ERROR: Requiero directorio del modelo Word2Vec')
    parser.exit(1)
  if not args.d:
    print('ERROR: Requiero directorio del dataset preparado')
    parser.exit(1)
  if not args.p:
    print('ERROR: Requiero directorio de salida del modelo preparado')
    parser.exit(1)

  #Carga el modelo de Word2Vec
  modelo=load_model(args.m)
  #Vectoriza los párrafos con Word2Vec
  data = word2vec(args.d,modelo)
  #Entrena el modelo SVM
  SVM(data[0],data[1],args.p)

  #Aplica la reducción de dimensionalidad 
  #con PCA
  X_1=redimencionarPCA(data[0])
  plot_points_with_labels(X_1,data[1])
  #conTSNE
  X_2=redimencionarTSNE(data[0])
  plot_points_with_labels(X_2,data[1])
  #conISOMAP
  X_3=redimensionarISOMAP(data[0])
  plot_points_with_labels(X_3,data[1])
  print('DONE')

if __name__ == '__main__':
  main()
