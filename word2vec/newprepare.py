
import gensim
import os
import re
import json as js
import argparse
from gensim.models import KeyedVectors
from gensim import models



def prepare_data_train(problemdir,prepareddir):

        if not os.path.exists(prepareddir):
            os.makedirs(prepareddir)

        f_train = open(os.path.join(prepareddir,'train.jsonl'),'w',encoding="utf-8")
        f_train_truth = open(os.path.join(prepareddir,'train_truth.jsonl'),'w',encoding="utf-8")

        for element in [x for x in os.listdir(problemdir) if '.txt' in x]:
            id = re.findall(r'^[a-zA-Z\-0-9]*\-([0-9]+).txt$',element)[0]
            with open(os.path.join(problemdir,f'problem-{id}.txt'),'r', encoding="utf-8") as f:
                data = f.readlines()
            with open(os.path.join(problemdir,f'truth-problem-{id}.json'),'r', encoding="utf-8") as f:
                changes = js.load(f)["changes"]
            numautor = 1
            data_labels = [numautor]
            for change in changes:
                if change == 1:
                    numautor+=1
                data_labels.append(numautor)
            if len(data_labels) != len(data):
                print(f"Descartando archivo con id {id}")
                continue

            for i in range(0,len(data)-1):
                js.dump({"id":f'{id}-{i+1}-{i+2}',"pair": [data[i],data[i+1]]}, f_train)
                f_train.write('\n')
                js.dump({"id":f'{id}-{i+1}-{i+2}',"diff": int(changes[i])}, f_train_truth)
                f_train_truth.write('\n')
        f_train.close()
        f_train_truth.close()
        print("Data for training done")

def main():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-d', type=str, help='Directorio del dataset')
  parser.add_argument('-p', type=str, help='Directorio del dataset preparado')
  args = parser.parse_args()

  if not args.d:
    print('ERROR: Requiero directorio del dataset')
    parser.exit(1)
  if not args.p:
    print('ERROR: Requiero directorio de saida de dataset preparado')
    parser.exit(1)

  prepare_data_train(args.d,args.p)

if __name__ == '__main__':
  main()