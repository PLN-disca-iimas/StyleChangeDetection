import argparse
import json as js
import os
import re

parser = argparse.ArgumentParser(description='Prepare data for train and validate model')
parser.add_argument('-r', type=str,
        help='Path to the data folder')
parser.add_argument('-y', type=int, default = 0, 
        help='Flag for determinate if the folder in train information or tes/validation information')
parser.add_argument('-o', type=str,
        help='Folder to save prepared data')
args = parser.parse_args()

if not args.r:
    raise ValueError('The train folder path is required')
if not args.o:
    raise ValueError('The output folder is required')

if args.y:
    f_train = open(os.path.join(args.o,'train.jsonl'),'a',encoding="utf-8")
    f_train_truth = open(os.path.join(args.o,'train_truth.jsonl'),'a',encoding="utf-8")
    for element in [x for x in os.listdir(args.r) if '.txt' in x]:
        id = re.findall(r'^[a-zA-Z\-0-9]*\-([0-9]+).txt$',element)[0]
        with open(os.path.join(args.r,f'problem-{id}.txt'),'r') as f:
            data = f.readlines()
        with open(os.path.join(args.r,f'truth-problem-{id}.json'),'r') as f:
            data_labels = js.load(f)["paragraph-authors"]

        for i in range(0,len(data)-1):
            for j in range(i+1,len(data)):
                js.dump({"id":f'{id}-{i+1}-{j+1}',"pair": [data[i],data[j]]}, f_train)
                f_train.write('\n')
                js.dump({"id":f'{id}-{i+1}-{j+1}',"same": int(data_labels[i] == data_labels[j])}, f_train_truth)
                f_train_truth.write('\n')
    f_train.close()
    f_train_truth.close()

else:
    data_parsed = []
    for element in [x for x in os.listdir(args.r) if '.txt' in x]:
        id = re.findall(r'^[a-zA-Z\-0-9]*\-([0-9]+).txt$',element)[0]
        with open(os.path.join(args.r,f'problem-{id}.txt'),'r') as f:
            data = f.readlines()
        with open(os.path.join(args.r,f'truth-problem-{id}.json'),'r') as f:
            data_labels = js.load(f)["paragraph-authors"]
        
        comb = []
        for i in range(0,len(data)-1):
            comb.append({"id":f'{id}-{i+1}-{i+2}',"pair": [data[i],data[i+1]]})
        data_parsed.append(comb)
    
    with open(os.path.join(args.o,'validation.json'),'w',encoding="utf-8") as f:
        js.dump(data_parsed, f, indent=4)