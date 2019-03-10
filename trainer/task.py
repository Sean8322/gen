# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import argparse
import datetime
import pprint
import subprocess
import keras
from keras import backend
from keras import optimizers
import argparse
import logging
import os
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

import numpy as np
# from . import model
# from . import utils

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
print('Libraries Imported')
# ps_hosts = FLAGS.ps_hosts.split(",")
# worker_hosts = FLAGS.worker_hosts.split(",")

# cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        /a/1.txt
        /a/b/2.txt

    If you just specify prefix = '/a', you'll get back:

        /a/1.txt
        /a/b/2.txt

    However, if you specify prefix='/a' and delimiter='/', you'll get back:

        /a/1.txt

    """
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


trainList = list_blobs_with_prefix('biobuck', 'arraylong/', delimiter='/')
# testList = list_blobs_with_prefix('oxygen-bac', 'bio/test/', delimiter='/')


flip = 1
def generator(listy, batch_size):
    global flip
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size,5000000,21))
    batch_labels = np.zeros((batch_size,1))
    # print("batch size is " + str(batch_size))
    while True:
     for i in range(0, batch_size):
       # batch_features.append([])
       # choose random index in features
       print("Length of listy is " + str(len(listy)))
       index = np.random.choice(len(listy),1)
       # print("preformatted index is" )
       # print(index)
       index = int(index)
       # print("cur index is ")
       # print(index)
       subprocess.check_call(['gsutil', '-q', 'cp', ('gs://biobuck/' + listy[index]), './'])
       print('gs://biobuck/' + listy[index])
       # print(os.listdir())
       file = open(listy[index][-14:], 'rb')
       batch_labels[i] = (bool(int(listy[index][19])))
       if batch_labels[i] == flip:
           i-=1
           print("Skip " + str(batch_labels[i]))
           continue
       else:
           print("Keep going " + str(batch_labels[i]))
           if flip == 1:
               flip = 0
           else:
               flip = 1
       temparray = (pickle.load(file, encoding='latin1'))
       # print(temparray[0])
       cattedTrain = keras.utils.to_categorical(temparray, num_classes=21, dtype = np.bool_)
       batch_features[i] = np.array(cattedTrain)
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
       subprocess.check_call(['rm', './'+listy[index][-14:]])
       # print(np.shape(batch_nump))
       print(batch_features.shape)
       print(batch_labels.shape)
       print(sys.getsizeof(batch_features))
       print(sys.getsizeof(batch_labels))
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

def get_args():
  """Argument parser.

	Returns:
	  Dictionary of arguments.
	"""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--job-dir',
    type=str,
    default='gs://biobuck/',
    help='GCS location to write checkpoints and export models')
  parser.add_argument(
    '--train-file',
    type=str,
    default='gs://biobuck/bio/train',
    help='Training file local or GCS')
  parser.add_argument(
    '--test-file',
    type=str,
    default='gs://biobuck/bio/test',
    help='Test file local or GCS')
  parser.add_argument(
    '--num-epochs',
    type=float,
    default=10,
    help='number of times to go through the data, default=5')
  parser.add_argument(
    '--batch-size',
    default=10,
    type=int,
    help='number of records to read during each training step, default=128')
  parser.add_argument(
    '--learning-rate',
    default=.01,
    type=float,
    help='learning rate for gradient descent, default=.001')
  parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='INFO')
  return parser.parse_args()


def train_and_evaluate(hparams):
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
  # for name in trainList:
  #     subprocess.check_call(['gsutil', '-q', 'cp', ('gs://oxygen-bac/' + name), './'])
  #     print('gs://oxygen-bac/' + name)
  #     print(os.listdir())
  #     file = open(name[-14:], 'rb')
  #     label.append(bool(int(name[19])))
  #     trainArr.append(pickle.load(file, encoding='latin1'))
  #     file.close()
  #     subprocess.check_call(['rm', './'+name[-14:]])
  # print("Loaded")
  # for name in testList:
  #     file = open(test_file+name)
  #     testArr.append(pickle.load(file))
  # x_train = np.array(trainArr[0:int((len(trainArr))/2)])
  # x_test = np.array(trainArr[int(len(trainArr)/2):int(len(trainArr))])
  # y_train = np.array(label[0:int((len(label))/2)])
  # y_test = np.array(label[int(len(label)/2):int(len(label))])
  # print("Data prepared")
  # print(x_train.shape())
  # print(y_train.shape())
  # print(x_test.shape())
  # print(y_test.shape())
  # Define training steps.
  # train_steps = hparams.num_epochs * len(
      # train_images) / hparams.batch_size
  # Create TrainSpec.
  # train_labels = np.asarray(train_labels).astype('int').reshape((-1, 1))
  # train_spec = tf.estimator.TrainSpec(
  #     input_fn=lambda: model.input_fn(
  #         train_images,
  #         train_labels,
  #         hparams.batch_size,
  #         mode=tf.estimator.ModeKeys.TRAIN),
  #     max_steps=train_steps)

  # Create EvalSpec.
  # exporter = tf.estimator.LatestExporter('exporter', model.serving_input_fn)
  # Shape numpy array.
  # eval_spec = tf.estimator.EvalSpec(
  #     input_fn=lambda: model.input_fn(
  #         test_images,
  #         test_labels,
  #         hparams.batch_size,
  #         mode=tf.estimator.ModeKeys.EVAL),
  #     steps=None,
  #     exporters=exporter,
  #     start_delay_secs=10,
  #     throttle_secs=10)



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
  model.add(Conv1D(400, kernel_size=(1), activation=tf.nn.sigmoid, input_shape = (5000000,21,)))
  model.add(layers.Dropout(0.3))
  # model.add(Conv1D(200000, kernel_size=(20), activation=tf.nn.relu))
  # model.add(Conv1D(500, kernel_size=(100), activation=tf.nn.sigmoid))
  model.add(Conv1D(70, kernel_size=(50), activation=tf.nn.relu))
  # model.add(Conv1D(1000, kernel_size=(2,2), activation=tf.nn.relu))÷
  # print(model.summary())
  model.add(layers.Dropout(0.2))
  model.add(Dense(90, activation=tf.nn.sigmoid))
  model.add(Dense(50, activation=tf.nn.sigmoid))
  model.add(Dense(10, activation=tf.nn.relu))
  model.add(layers.Dropout(0.2))
  model.add(Flatten())
  # model.add(Dense(100, activation=tf.nn.relu))
  # model.add(layers.Dropout(0.2))

  # model.add(Flatten())
  # model.add(Dense(10, activation=tf.nn.relu))
  model.add(BatchNormalization())
  model.add(Dense(1, activation=tf.nn.sigmoid))
  print(model.summary())


  # Compile model with learning parameters.
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  sgd = optimizers.SGD(lr=0.15, decay=1e-6, momentum=0.9, nesterov=True)

  model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

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
  print(str(get_model_memory_usage(1, model))+ "GB Required")
  model.fit_generator(generator(trainList[0:int(len(trainList)/2)], 1), steps_per_epoch=7, epochs=10)
  loss_and_metrics = model.evaluate_generator(generator(trainList[int(len(trainList)/2):int(len(trainList))], 10), steps=25, verbose=1)

  # file = open('metrics', "w+")
  # file.write(str(loss_and_metrics[0]))
  # file.write('\n')
  # file.write(str(loss_and_metrics[1]))
  # file.close()

  print(loss_and_metrics)
  model.save("oxygen.h5")

  subprocess.check_call(['gsutil', 'cp', './oxygen.h5', 'gs://biobuck/'])
if __name__ == '__main__':
  print("Running")
  args = get_args()
  # tf.logging.set_verbosity(args.verbosity)
  hparams = hparam.HParams(**args.__dict__)
  train_and_evaluate(hparams)
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
