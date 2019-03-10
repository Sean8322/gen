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

import argparse
import datetime
import pprint
import subprocess
from tensorflow import keras
from keras import backend
from keras import optimizers
import argparse
import logging
import os
import pickle
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras import layers

import numpy as np
# from . import model
# from . import utils

import tensorflow as tf

print('Libraries Imported')

def generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, 6000, 5000, 21))
    batch_labels = np.zeros((batch_size,1))
    while True:
     for i in range(batch_size):
       # choose random index in features
       index= np.random.choice(len(features),1)
       batch_features[i] = features[index]
       batch_labels[i] = labels[index]
     yield batch_features, batch_labels
trainList = os.listdir('array')
# testList = list_blobs_with_prefix('oxygen-bac', 'bio/test/', delimiter='/')

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
  for name in trainList:
      print('array/' + name)
      # print(os.listdir())
      # exit()
      file = open('array/' + name, 'rb')
      label.append(bool(int(name[9])))
      trainArr.append(pickle.load(file, encoding='latin1'))
      file.close()
  print("Loaded")
  # for name in testList:
  #     file = open(test_file+name)
  #     testArr.append(pickle.load(file))
  x_train = np.array(trainArr[0:int((len(trainArr))/2)])
  x_test = np.array(trainArr[int(len(trainArr)/2):int(len(trainArr))])
  y_train = np.array(label[0:int((len(label))/2)])
  y_test = np.array(label[int(len(label)/2):int(len(label))])
  print("Data prepared")
  print(x_train.shape())
  print(y_train.shape())
  print(x_test.shape())
  print(y_test.shape())
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
  model.add(Conv2D(100, kernel_size=(1,1), activation=tf.nn.relu, input_shape = (6000,5000,21)))
  # model.add(Conv1D(500000, kernel_size=(3), activation=tf.nn.relu))
  # model.add(Conv1D(250000, kernel_size=(3), activation=tf.nn.sigmoid))
  # model.add(Conv2D(300, kernel_size=(1,2), activation=tf.nn.relu))
  # model.add(Conv2D(100, kernel_size=(2,2), activation=tf.nn.relu))
  print(model.summary())

  # model.add(Dense(500000, activation=tf.nn.relu))
  # model.add(Dense(50000, activation=tf.nn.sigmoid))
  model.add(Flatten())
  model.add(Dense(100, activation=tf.nn.relu))
  model.add(layers.Dropout(0.5))
  model.add(Dense(50, activation=tf.nn.relu))
  # model.add(Flatten())
  # model.add(Dense(10, activation=tf.nn.relu))
  model.add(Dense(1, activation=tf.nn.relu))
  print(model.summary())


  # Compile model with learning parameters.
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

  model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])


  model.fit_generator(generator(x_train, y_train, 1), steps_per_epoch=50, epochs=10)
  loss_and_metrics = model.evaluate_generator(generator(x_train, y_train, 1), steps=20, verbose=1)

  # file = open('metrics', "w+")
  # file.write(str(loss_and_metrics[0]))
  # file.write('\n')
  # file.write(str(loss_and_metrics[1]))
  # file.close()

  print(loss_and_metrics)
  model.save("oxygen.h5")

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
