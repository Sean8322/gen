import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from ann_visualizer.visualize import ann_viz

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


trainList.sort()
testList.sort()
print(len(trainList))
a = trainList[0:42]
b = trainList[42:84]
c = trainList[84:126]
# print(int((len(trainList))/3), int((len(trainList)+1)/3), int((2*len(trainList))/3), len(trainList))
print(len(a), len(b), len(c))
hold = []
for i in range(0, int(len(trainList)/3)):
    print(i)
    hold.append(a[i])
    hold.append(b[i])
    hold.append(c[i])
trainList = hold

print(len(trainList), trainList)
print(len(testList), testList)
# testList = list_blobs_with_prefix('oxygen-bac', 'bio/test/', delimiter='/')


flip = 2
def generator(path, listy):
    index = 0

    while True:
        global flip
        if index == 125:
            index = 0
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
        print("Is", lab)
        batch_labels = np.full((len(cattedTrain),1), lab)
        # print("labels are", batch_labels)
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
def get_model_memory_usage(batch_size, model):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
         print("Small Float")
    if K.floatx() == 'float64':
         number_size = 8.0
         print("Big Float")
    else:
        print("Something else " + str(type(K.floatx())))

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


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
  # run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)
  # Create estimator.
  # estimator = model.keras_estimator(
  #   model_dir=hparams.job_dir,
  #   config=run_config,
  #   learning_rate=hparams.learning_rate)
  # Start training
  print("Starting training now")
  # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  # estimator.export_saved_model('saved_model', serving_input_receiver_fn)
  model = Sequential()
  model.add(Conv1D(81, kernel_size=(15), activation=tf.nn.sigmoid, input_shape = (chunk,21,)))
#   model.add(layers.Dropout(0.3))
#   model.add(Conv1D(1000, kernel_size=(20), activation=tf.nn.relu))
  model.add(Conv1D(27, kernel_size=(10), activation=tf.nn.sigmoid))
#   model.add(Conv1D(100, kernel_size=(50), activation=tf.nn.relu))
  
  # model.add(Conv1D(1000, kernel_size=(2,2), activation=tf.nn.relu))÷
  # print(model.summary())
#   model.add(layers.Dropout(0.2))
  model.add(BatchNormalization())
#   model.add(Dense(90, activation=tf.nn.sigmoid))
#   model.add(Dense(50, activation=tf.nn.sigmoid))
#   model.add(layers.Dropout(0.2))
  model.add(Flatten())
  # model.add(Dense(100, activation=tf.nn.relu))
  # model.add(layers.Dropout(0.2))

  # model.add(Flatten())
  # model.add(Dense(10, activation=tf.nn.relu))
  
#   model.add(Dense(10, activation=tf.nn.relu))
  model.add(Dense(3, activation=tf.nn.sigmoid))
  print(model.summary())


  # Compile model with learning parameters.
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


  model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#   ann_viz(model, title="", filename = "viz.gv")
  # strategy = tf.contrib.distribute.MirroredStrategy()
  # config = tf.estimator.RunConfig(train_distribute=strategy)
  # est = tf.keras.estimator.model_to_estimator(model, config=config)
  # train_spec = tf.estimator.TrainSpec(input_fn=generator(trainList[0:int(len(trainList)/2)], 1), max_steps=1000)
  # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
  # tf.estimator.train_and_evaluate(
  #   est,
  #   train_spec,
  #   eval_spec
  # )
  print(str(get_model_memory_usage(200, model))+ "GB Required")
  plot_model(model, to_file='model.png', show_shapes=True)
#   subprocess.check_call(['gsutil', 'cp', './model.png', 'gs://biobuck/'])
  history = model.fit_generator(generator("newtrain/", trainList), steps_per_epoch=1, epochs=126, max_queue_size = 1)
  loss_and_metrics = model.evaluate_generator(generator("newtest/", testList), steps=100, verbose=1, max_queue_size = 1)
  model.save("oxygen2.h5")
  hist = open("trainingdata", 'wb')
  pickle.dump(history, hist)
  print(history.history.keys())
  
  # file = open('metrics', "w+")
  # file.write(str(loss_and_metrics[0]))
  # file.write('\n')
  # file.write(str(loss_and_metrics[1]))
  # file.close()

  print(loss_and_metrics)
  model.save("oxygen2.h5")

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
