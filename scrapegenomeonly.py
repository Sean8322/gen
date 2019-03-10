from Bio import Entrez
import time
import re
from tqdm import tqdm
import codecs
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras import layers
import gc
import pickle
zeroes = [0] * 8000
trainingSetSize = 1003
testingSetSize = 97
linesToRead = 1003
Entrez.email = "seacheo@nuevaschool.org"
Entrez.api_key = "c268992545656cfcb41217f8583361c99707"
genids = open("genbankid.txt", "r")
location = 0
maxl = 0
p = 0
cattedTrain = []
gc.enable()
def gen(x):
    return {
        'c': 1,
        'g': 2,
        'a': 3,
        't': 4,
        'n': 0,
    }[x]
k = 0
j = 0
label = []
curid = []
raw = []
fast = []
for p in tqdm(range(0, linesToRead)):
    usefulGenesStart = []
    usefulGenesEnd = []
    tempcur = genids.readline()
    if (tempcur[0] == 'C' and tempcur[1] == 'P'):
        curid.append(tempcur)
        
        try: 
            handle = (Entrez.efetch(db="nucleotide", id=curid[j], rettype="gb", retmode="xml"))
        except:
            print("Some random error nucleotide")
            continue
        try:
            fasta =(Entrez.efetch(db="nucleotide", id=curid[j], rettype="fasta", retmode="text"))
        except:
            print("Some random error fasta")
            continue
        # file = codecs.open('dump-genome/' + curid[j][:-1] +'.txt', "w+", "utf-8")
        raw.append(Entrez.read(handle))
        

        j+=1
print("Finished fetching from NCBI Server")
print(j)
# print(len(handle))
# print(len(fasta))
# print(type(handle))
# print((handle[0]))
# print((handle[1]))
# for k in (range(0, j)):
#     print(Entrez.read(handle[k]))
#
# exit()
for g in (range(0, j)):
    curOxy = 0
    array = []
    print("Start decode " + str(g))
    print("Decoded raw Aerobe file")
    wholegenome = raw[g][0]["GBSeq_sequence"]
    # print(wholegenome)
    if (len(wholegenome) < 1000000):
        print("genome too short")
        continue
    
    print("Opened Genome file")

    # print(raw)
    comments = raw[g][0]["GBSeq_comment"]
    if (comments.find("naerobe") != -1):
        print(str(curid[g]) + " is an Anaerobe")
        # file.write("2\n")
        label.append(1)
        curOxy = 2
    elif (comments.find("erobe") != -1 or comments.find("icroaerophilic") != -1):
        print(str(curid[g]) + "is an Aerobe")
        # file.write("0\n")
        label.append(0)
        curOxy = 0
    elif (comments.find("acultative") != -1):
        print(str(curid[g]) + " can change")
        # file.write("1\n")
        label.append(1)
        curOxy=1
    else:
        print(curid[g] + "Error no oxygen relation found")
        print(comments)
        # file.close()
        continue
        # exit(1)
    file = open('dump-genome/' + curid[g][:-1] +'_'+str(curOxy), "wb")
    
    arr = []
    for c in wholegenome:
        arr.append(gen(c))
    # print(wholegenome)
    # file.write(wholegenome)
    # file.close()

    

    print("finished " + str(g))
    # print(len(array[0]))
    # print((array[0]))
    # print(len(array[1]))
    # print((array[1]))
    # for k in range(0, len(array)):
    #     for j in range (len(array[k]), 8000):
    #         array[k].append(0)
    # print(len(array))
    # for k in range(len(array), 7000):
    #     array.append(zeroes)
    # print("Finished padding")
    # print(len(array))
    # if(len(array) == 0):
    #     print("Too short")
    #     continue
    # print(len(array[0]))
    # print((array[0]))
    # print(len(array[1]))
    # print((array[1]))
    # exit(0)
    # for k in range(0, 6000):
    #     # print("Protein chain " + str(k) + " on genome " + str(i))
    #     cattedTrain.append([])
    #     for j in range(0, 5000):
    #         cattedTrain[k].append(keras.utils.to_categorical(array[k][j]))
    # print("Finished converting to categorical")
    # def Sorting(lst): 
    #     lst.sort(key=len) 
    #     return lst 
    # array = Sorting(array)
    pickle.dump(arr, file)
    cattedTrain = []
    print("Saved")
