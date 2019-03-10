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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.python.keras import models

tf.logging.set_verbosity(tf.logging.INFO)


def keras_estimator(model_dir, config, learning_rate):
  """Creates a Keras Sequential model with layers.

  Args:
    model_dir: (str) file path where training files will be written.
    config: (tf.estimator.RunConfig) Configuration options to save model.
    learning_rate: (int) Learning rate.

  Returns:
    A keras.Model
  """
  model = models.Sequential()
  model.add(Conv1D(1500000, kernel_size=(4), activation=tf.nn.relu, input_shape = (15000000,6)))
  # model.add(Conv1D(5000000, kernel_size=(4), activation=tf.nn.relu))
  model.add(Conv1D(250000, kernel_size=(3), activation=tf.nn.sigmoid))
  model.add(Dense(200000, activation=tf.nn.relu))
  model.add(Conv1D(1000, kernel_size=(2), activation=tf.nn.relu))
  # model.add(Dense(500000, activation=tf.nn.relu))
  # model.add(Dense(50000, activation=tf.nn.sigmoid))
  model.add(Dense(5000, activation=tf.nn.relu))
  model.add(Dense(50, activation=tf.nn.relu))
  model.add(Dense(1, activation=tf.nn.relu))

  # Compile model with learning parameters.
  # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

  model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

  # estimator = tf.keras.estimator.model_to_estimator(
  #     keras_model=model, model_dir=model_dir, config=config)
  return model


def input_fn(features, labels, batch_size, mode):
  """Input function.

  Args:
    features: (numpy.array) Training or eval data.
    labels: (numpy.array) Labels for training or eval data.
    batch_size: (int)
    mode: tf.estimator.ModeKeys mode

  Returns:
    A tf.estimator.
  """
  # Default settings for training.
  if labels is None:
    inputs = features
  else:
    # Change numpy array shape.
    inputs = (features, labels)
  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices(inputs)
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
    dataset = dataset.batch(batch_size)
  return dataset.make_one_shot_iterator().get_next()


def serving_input_fn():
  """Defines the features to be passed to the model during inference.

  Expects already tokenized and padded representation of sentences

  Returns:
    A tf.estimator.export.ServingInputReceiver
  """
  feature_placeholder = tf.placeholder(tf.float32, [None, 784])
  features = feature_placeholder
  return tf.estimator.export.TensorServingInputReceiver(features,
                                                        feature_placeholder)
