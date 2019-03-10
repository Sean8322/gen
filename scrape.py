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
def acid(x):
    return {
        'A': 1,
        'R': 2,
        'N': 3,
        'D': 4,
        'C': 5,
        'Q': 6,
        'E': 7,
        'G': 8,
        'H': 9,
        'I': 10,
        'L': 11,
        'K': 12,
        'M': 13,
        'F': 14,
        'P': 15,
        'S': 16,
        'T': 17,
        'W': 18,
        'Y': 19,
        'V': 20,
        'X': 0,
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
        file = codecs.open('dump-genome/' + curid[j][:-1] +'.txt', "w+", "utf-8")
        raw.append(Entrez.read(handle))
        # try:
        #     Entrez.read(handle[j])
        # except:
        #     print("Ruh roh NCBI gene read didnt work " + str(j) + " " + curid)
        #     handle[j] = (Entrez.efetch(db="nucleotide", id=curid[j], rettype="gb", retmode="xml"))
        
        fast.append(fasta.read())
        # try:
        #     fasta[j].read()
        # except:
        #     print("Ruh roh NCBI fasta read didnt work " + str(j) + " " + curid)
        #     fasta[j] = (Entrez.efetch(db="nucleotide", id=curid[j], rettype="fasta", retmode="text"))
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
    file = codecs.open('dump-genome/' + curid[g][:-1] +'.txt', "w+", "utf-8")
    print("Opened Genome file")

    # print(raw)
    comments = raw[g][0]["GBSeq_comment"]
    if (comments.find("naerobe") != -1):
        print(str(curid[g]) + " is an Anaerobe")
        file.write("2\n")
        label.append(1)
        curOxy = 2
    elif (comments.find("erobe") != -1 or comments.find("icroaerophilic") != -1):
        print(str(curid[g]) + "is an Aerobe")
        file.write("0\n")
        label.append(0)
        curOxy = 0
    elif (comments.find("acultative") != -1):
        print(str(curid[g]) + " can change")
        file.write("1\n")
        label.append(1)
        curOxy=1
    else:
        print(curid[g] + "Error no oxygen relation found")
        print(comments)
        file.close()
        continue
        # exit(1)

    wholegenome = raw[g][0]["GBSeq_sequence"]
    # print(wholegenome)
    file.write(wholegenome)
    file.close()

    # print(fasta.read())
    temp = codecs.open('tempgen.fna', "w+", "utf-8")
    temp.write(fast[g])
    temp.close()
    print("Start protein finder")
    subprocess.run(["prodigal", "-q", "-i", "tempgen.fna", "-a", 'dump-protein/' + curid[g][:-1] +'.faa'])
    # if (subprocess.check_call(["prodigal", "-q", "-i", "tempgen.fna", "-a", 'dump-protein/' + curid[g][:-1] +'.faa']) != 0):
    #     print("Error in prodigal")
    #     exit(1)
    print("Protein locations found")
    string = []
    i = 0
    protein = codecs.open('dump-protein/' + curid[g][:-1] +'.faa', "r", "utf-8")
    line = protein.readline()
    n=-1
    while line:
        # print(line[len(line)-2])
        # print(line)
        if (line[0] != '>'):
            for c in line:
                if (c >= "A" and c <= "Z"):
                    # print(c + str(acid(c)))
                    # print(n)
                    array[n].append(acid(c))
                    # print('i = ' + str(i))

                # print(line)
            # print(line)
        else:
            # print(n)
            n+=1
            array.append([])
        line = protein.readline()

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
    print(len(array))
    if(len(array) == 0):
        print("Too short")
        continue
    print(len(array[0]))
    # print((array[0]))
    print(len(array[1]))
    # print((array[1]))
    # exit(0)
    # for k in range(0, 6000):
    #     # print("Protein chain " + str(k) + " on genome " + str(i))
    #     cattedTrain.append([])
    #     for j in range(0, 5000):
    #         cattedTrain[k].append(keras.utils.to_categorical(array[k][j]))
    # print("Finished converting to categorical")
    def Sorting(lst): 
        lst.sort(key=len) 
        return lst 
    array = Sorting(array)
    with open("array/" + curid[g][:-1] + "_" + str(curOxy)+ ".txt", "wb") as fp:
        pickle.dump(array, fp)
    cattedTrain = []
    print("Saved")

    # print(array[k])
    # os.remove("temp.fna")
    # k+=1
    # print(array)
    # exit()
    # meta = raw[0]["GBSeq_feature-table"]
    # print (len(meta))
    # file.write(str(len(meta)-2))
    # file.write("\n")

    # print(raw[0])
    # k=0
    # print(len(meta))
    # offset = 0
    # for i in tqdm(range(0, len(wholegenome))):
    #     # print(wholegenome[i])
    #     # print(feature)
    #     if (wholegenome[i] == 'a'):
    #         file.write(str(1))
    #         curlen+=1
    #     elif (wholegenome[i] == 't'):
    #         file.write(str(2))
    #         curlen+=1
    #     elif (wholegenome[i] == 'g'):
    #         file.write(str(3))
    #         curlen+=1
    #     elif (wholegenome[i] == 'c'):
    #         file.write(str(4))
    #         curlen+=1
    #     elif (wholegenome[i] == 'n'):
    #         file.write(str(0))
    #         curlen+=1
    #     else:
    #         print("unknown sequence")
    #         print(wholegenome[k])
    #         exit(0)

        # curlen+=1
    # file.write(wholegenome[(int(usefulGenesStart[i])-1):(int(usefulGenesEnd[i])-1)])
    # file.write("\n")
    # print(curlen)
    # if (curlen > maxl):
    #     maxl = curlen
    # print("max " + str(maxl))
    # pad = '0'
    # for g in range(curlen+1, 2000000):
    #     pad = pad + '0'
    # file.write(pad)
    # print(usefulGenesStart[1])
    # print(usefulGenesEnd[1])
    # file.write(wholegenome)
    # for match in re.finditer('gene', wholegenome):
    #     print (match.end())
# print("Array loaded")
# zeroes = [0] * 5000
# cattedTrain = []
# for i in range(0, len(array)):
#     for k in range(0, len(array[i])):
#         for j in range (len(array[i][k]), 5000):
#             array[i][k].append([0])
#     for k in range(len(array[i]), 6000):
#         array[i].append(zeroes)
# print("Finished padding")
# for i in range(0, len(array)):
#     print("Starting categorical " + str(i))
#     cattedTrain.append([])
#     for k in range(0, 6000):
#         # print("Protein chain " + str(k) + " on genome " + str(i))
#         cattedTrain[i].append([])
#         for j in range(0, 5000):
#             cattedTrain[i][k].append(keras.utils.to_categorical(array[i][k][j]))

#
# print(type(cattedTrain))
# print(type(cattedTrain[0]))
# print(type(cattedTrain[0][0]))
# print(len(cattedTrain))
# print(len(cattedTrain[0]))
# print(len(cattedTrain[0][0]))
# print(len(cattedTrain[0][1]))
# print(len(cattedTrain[0][0][0]))
# print(len(cattedTrain[0:int((len(cattedTrain))/2)]))
# print(len(cattedTrain[int(len(cattedTrain)/2):(len(cattedTrain))]))
# print(len(label[0:int((len(label))/2)]))
# print(len(label[int(len(label)/2):int(len(label))]))
# x_train = np.array(cattedTrain[0:int((len(cattedTrain))/2)])
# x_test = np.array(cattedTrain[int(len(cattedTrain)/2):(len(cattedTrain))])
# y_train = np.array(label[0:int((len(label))/2)])
# y_test = np.array(label[int(len(label)/2):int(len(label))])
#
# with open("trainList.txt", "wb") as fp:
#     pickle.dump(cattedTrain, fp)
# with open("trainLabel.txt", "wb") as fpd:
#     pickle.dump(label, fpd)
#
# np.save('x_train.npy', np.array(cattedTrain[0:int((len(cattedTrain))/2)]))
# np.save('x_test.npy', np.array(cattedTrain[int(len(cattedTrain)/2):(len(cattedTrain))]))
# np.save('y_train.npy', np.array(label[0:int((len(label))/2)]))
# np.save('y_test.npy', np.array(label[int(len(label)/2):int(len(label))]))
# print("Into numpy array")
# del cattedTrain[:]
# print("deleted")
# gc.collect(generation=2)
# print("collected")
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# np.save('x_train.npy', x_train)
# np.save('x_test.npy', x_test)
# np.save('y_train.npy', y_train)
# np.save('y_test.npy', y_test)

# print("Saved")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train,
#                                                         value=0,
#                                                         padding='post',
#                                                         maxlen=5000)
#
# x_test = keras.preprocessing.sequence.pad_sequences(x_test,
#                                                         value=0,
#                                                         padding='post',
#                                                         maxlen=5000)
