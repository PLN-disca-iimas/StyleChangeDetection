import json as js
import os
import re

BASE_DIR = 'pan22/dataset1/validation/' 
index = 0
for element in os.listdir(BASE_DIR):
    if '.txt' in element:        
        id = re.findall(r'^[a-zA-Z\-0-9]*\-([0-9]+).txt$',element)[0]
        with open(os.path.join(BASE_DIR,f'problem-{id}.txt'),'r') as f:
            data = f.readlines()
        with open(os.path.join(BASE_DIR,f'truth-problem-{id}.json'),'r') as f:
            data_labels = js.load(f)["paragraph-authors"]
        comb = []

        for i in range(0,len(data)-1):
            for j in range(i+1,len(data)):
                with open('validation.jsonl','a',encoding="utf-8") as f:
                    js.dump({"id":f'{id}-{i}-{index}',"pair": [data[i],data[j]]}, f)
                    f.write('\n')
                with open('validation_truth.jsonl','a',encoding="utf-8") as f:
                    js.dump({"id":f'{id}-{i}-{index}',"same": data_labels[i] == data_labels[j]}, f)
                    f.write('\n')
                index+=1 