import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import load_model

import argparse
import datetime
import pprint
import subprocess
import keras
from keras import backend
from keras import optimizers
import argparse
import logging
import pickle
import sys
from google.cloud import storage
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.utils.training_utils import multi_gpu_model
from keras import layers
import keras.backend as K
from keras.utils import plot_model
import random
import numpy as np
# import matplotlib.pyplot as plt

# from . import model
# from . import utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
chunk = 1000
print('Libraries Imported')
# ps_hosts = FLAGS.ps_hosts.split(",")
# worker_hosts = FLAGS.worker_hosts.split(",")

# cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)
    # return (blobs.prefixes)
    tempPref = []
    print('Blobs:')
    for blob in blobs:
        print(blob.name)
        tempPref.append(blob.name)
    # if delimiter:
    #     print('Prefixes:')
    #     for prefix in blobs.prefixes:
    #         print(prefix)
    #
    return(tempPref)


trainList = os.listdir("newtrain")
testList = os.listdir("newtest")

# def comp(val):
#     return int(val[:-1])
# def first(val):
#     return int(val[-1:])
# trainList.sort(key=first)
# testList.sort(key=first)
# trainList.sort(key=comp)
# testList.sort(key=comp)
print(trainList)
print(testList)
# testList = list_blobs_with_prefix('oxygen-bac', 'bio/test/', delimiter='/')


flip = 1
def generator(path, listy):
    index = 0

    while True:
        global flip
        if index == 125:
            break
        # Create empty arrays to contain batch of features and labels#
        # batch_features = np.zeros((batch_size,5000000,21))
        
        # print("batch size is " + str(batch_size))
        # batch_features.append([])
        # choose random index in features
        print("Length of listy is " + str(len(listy)))
        
        # while True:
            
        #     # print("preformatted index is" )
        #     # print(index)
        #     index = int(index)
        #     lab = int(listy[index][9])
        #     if lab != flip:
        #         print("Keep going " + str(lab))
        #         if flip == 1:
        #             flip = random.choice([0, 2])
        #         elif flip == 0:
        #             flip = random.choice([1,2])
        #         else:
        #             flip = random.choice([0,1])
        #         break
        print("index is", index, "len of listy", len(listy))
        lab = int(listy[index][0])
        # print("cur index is ")
        # print(index)
        # subprocess.check_call(['gsutil', '-q', 'cp', ('gs://biobuck/' + listy[index]), './'])
        print(listy[index])
        # print(os.listdir())
        file = open(path+listy[index], 'rb')
        # del listy[index]
        
        temparray = (pickle.load(file, encoding='latin1'))
        print(len(temparray))
        cattedTrain = []
        x = 0
        # for i in range(0, len(temparray)):
            # print("before", i, len(temparray[i]))
        for line in temparray:
            
            interm = []
            if (len(line) > chunk):
                # print("x is",x)
                # print("len is", len(line))
                # pre = line[:chunk]
                # while len(line)>1000:
                #     interm.append(line[:chunk])
                #     line = line[chunk:]
                # interm[len(interm)-1].append(0)
                # interm[len(interm)-1] = interm[len(interm)-1]+pre
                # interm[len(interm)-1] = interm[len(interm)-1][:1000]
                # print("interm shape is", (np.array(interm)).shape)
                # try:
                #     print("prev", len(interm[x-1]))
                #     print("norm", len(interm[x]))
                #     print("add", len(interm[x+1]))
                # except:
                #     print("Some index error")
                # # del temparray[x-1]
                # temparray = temparray[:x-2]+interm+temparray[x:]
                temparray[x] = temparray[x][:chunk]
            x+=1
        # for i in range(0, len(temparray)):
            # print("after", i, len(temparray[i]))
        print((np.array(temparray)).shape)
        print("Done wrapping")
        for protein in temparray:
            # print(len(protein))
            if (len(protein) < chunk):
                add = protein
                while len(protein) < chunk:
                    protein.append(0)
                    protein+= add
                protein = protein[:chunk]
                # print("multiply", len(protein))
            # else:
                # print(len(protein))
            cattedTrain.append(keras.utils.to_categorical(protein, num_classes=21))
        print("Is", lab, "bacteria", listy[index])
        batch_labels = np.full((len(cattedTrain),1), lab)
        batch_labels = keras.utils.to_categorical(batch_labels, num_classes=3)
        batch_features = np.array(cattedTrain)
        # print("Finished converting to categorical")

        # print(len(batch_features[0][0][0][0]))
        # print(np.size(batch_features))
        # print(type(batch_features))
        # batch_nump = np.array(batch_features)

        # print(type(batch_nump))
        # print(np.shape(batch_nump))
        # print(batch_nump[0][0][0][0])
        # print(type(batch_nump[0][0][0]))
        file.close()
        # print(np.shape(batch_nump))
        print(len(batch_features[0]))
        print(len(batch_features[1]))
        print(len(batch_features[len(batch_features)-1]))
        print(batch_features.shape)
        print(batch_labels.shape)
        print(sys.getsizeof(batch_features))
        print(sys.getsizeof(batch_labels))
        index +=1
        yield (batch_features, batch_labels)


def train_and_evaluate():
  """Helper function: Trains and evaluates model.

  Args:
    hparams: (dict) Command line parameters passed from task.py
  """
  # Loads data.
  # x_train, y_train, x_test, y_test = utils.prepare_data(train_file=hparams.job_dir, test_file=hparams.job_dir,)
  testArr = []
  trainArr = []
  label = []
  print("Loading")
  


  print("Running Config")
  # Define running config.
  model = load_model('oxygen2.h5')
  i = 0
  for inp in generator("newtest/", testList):
      print(model.predict(inp[0]))
      if(i == 3):
        break
      i+=1
#   plot_model(model, to_file='model.png', show_shapes=True, dpi=300)
#   subprocess.check_call(['gsutil', 'cp', './model.png', 'gs://biobuck/'])

  


if __name__ == '__main__':
  print("Running")
  # tf.logging.set_verbosity(args.verbosity)
  train_and_evaluate()
else:
    print("__name__ != main")
# def serving_input_receiver_fn():
#     """Serving input_fn that builds features from placeholders
#
#     Returns
#     -------
#     tf.estimator.export.ServingInputReceiver
#     """
#     number = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='number')
#     receiver_tensors = {'number': number}
#     features = tf.tile(number, multiples=[1, 2])
#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
