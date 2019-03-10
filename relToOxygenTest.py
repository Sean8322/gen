
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import optimizers
import sys
import os
# Helper libraries
import numpy as np
# import matplotlib.pyplot as plt
import subprocess
import codecs
import urllib.request

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras import layers

from keras.models import load_model

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

model = load_model(sys.path[0] + '/oxygen.h5')

# path = sys.path[0] + '/CP003368.txt '
def predict():
    urllib.request.urlretrieve(path, "genome.fna")

    print("Start protein finder")
    subprocess.run(["prodigal", "-q", "-i", "genome.fna", "-a", "temp.faa"])
    print("Protein locations found")
    protein = codecs.open("temp.faa", "r", "utf-8")
    line = protein.readline()
    n=-1
    array = []
    while line:
        # print(line[len(line)-2])
        # print(line)
        if (line[0] != '>'):
            for c in line:
                if (c >= "A" and c <= "Z"):
                    # print(c + str(acid(c)))
                    # print(n)
                    array.append(acid(c))
                    # print('i = ' + str(i))

                # print(line)
            # print(line)
        else:
            # print(n)
            n+=1
            array.append(0)
        line = protein.readline()

    for k in range(len(array), 5000000):
            array.append(0)
    print("Finished padding")
    cattedTrain = keras.utils.to_categorical(array, num_classes=21, dtype = np.bool_)
    arr = []
    arr.append(cattedTrain)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #
    # model.add(Dense(units=64, activation='relu', input_shape=(2300000,)))
    # model.add(Dense(48, activation='sigmoid'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(units=10, activation='softmax'))
    # model.add(Dense(1))
    # model.compile(loss='binary_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy'])

    out = model.predict(generator(trainList[0:int(len(trainList))], 1))
    print(out)
    if (out[0] == 1):
        print("Anaerobe")
    elif (out[0] == 0):
        print("Aerobe")
while True:
    path = input("Please paste link to fasta file\n")
    predict()
