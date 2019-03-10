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
import os
# [START storage_upload_file]
from google.cloud import storage
import subprocess

WORKING_DIR = os.getcwd()
# os.makedirs('test')
# os.makedirs('train')

# def download_files_from_gcs(source, destination):
#   """Download files from GCS to a WORKING_DIR/.
#   Args:
#     source: GCS path to the training data
#     destination: GCS path to the validation data.
#   Returns:
#     A list to the local data paths where the data is downloaded.
#   """
#   local_file_names = [destination]
#   gcs_input_paths = [source]
#   print(destination)
#   print(local_file_names)
#
#   # Copy raw files from GCS into local path.
#   print("Working Dir")
#   print(WORKING_DIR)
#   for local_file_name in local_file_names[0]:
#       print("Local")
#       print(local_file_name)
#       # raw_local_files_data_paths = [os.path.join(WORKING_DIR, local_file_name)]
#       # raw_local_files_data_paths = loca
#   raw_local_files_data_paths=local_file_names[0]
#   for i, gcs_input_path in enumerate(gcs_input_paths):
#     if gcs_input_path:
#       print(gcs_input_path)
#       print(local_file_names[0][i])
#       tempRemPath = gcs_input_path+local_file_names[0][i]
#       print(tempRemPath)
#       subprocess.check_call(
#         ['gsutil', 'cp', tempRemPath, raw_local_files_data_paths[0][i]])
#
#   return raw_local_files_data_paths

def prepare_data(train_file, test_file):
  # TensorFlow and tf.keras
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import optimizers
  import sys
  import os
  # from tqdm import tqdm
  # Helper libraries
  import numpy as np
  # import matplotlib.pyplot as plt
  # from tensorflow.keras import layers
  # from tensorflow.keras.models import Sequential
  # from tensorflow.keras.layers import Dense
  # from tensorflow.keras.layers import Flatten
  # from tensorflow.keras.layers import Conv1D
  #

  params = {'dim': (32,32,32),
            'batch_size': 2,
            'n_classes': 6,
            'n_channels': 1,
            'shuffle': True}

  trainData = []
  trainLabel = []
  testData = []
  testLabel = []

  trainingSetSize = 98
  testingSetSize = 98

  # x_train = np.array()
  # y_train = np.array()
  # x_test = np.array()
  # y_test = np.array()

  #--------------------------------------

  def generator(features, labels, batch_size):
   # Create empty arrays to contain batch of features and labels#
   batch_features = np.zeros((batch_size, 15000000, 5))
   batch_labels = np.zeros((batch_size,1))
   while True:
     for i in range(batch_size):
       # choose random index in features
       index= np.random.choice(len(features),1)
       batch_features[i] = features[index]
       batch_labels[i] = labels[index]
     yield batch_features, batch_labels
  #--------------------------------------


  # trainList = os.listdir(train_file)
  # testList = os.listdir(test_file)
  trainList = list_blobs_with_prefix('bac-oxygen', 'bio/train/', delimiter='/')
  testList = list_blobs_with_prefix('bac-oxygen', 'bio/test/', delimiter='/')
  # print("Lists:")
  # print(trainList)
  # print(trainData)
  # rawTrainPaths = download_files_from_gcs(train_file, destination = 'bio/train')
  # rawTestPaths = download_files_from_gcs(test_file, destination = 'bio/test')
  # trainList.remove('.DS_Store')
  # testList.remove('.DS_Store')
  print("downloading")
  # os.chdir(WORKING_DIR + '/train')
  subprocess.check_call(['gsutil', '-q', 'cp', '-r', (train_file + 'bio/train/'), './'])
  # os.chdir(WORKING_DIR + '/test')
  subprocess.check_call(['gsutil', '-q', 'cp', '-r', (test_file + 'bio/test/'), './'])
  # os.chdir(WORKING_DIR)
  print("downloaded")
  print(os.getcwd())
  print(os.listdir('/user_dir/test'))
  print(os.listdir('/user_dir/train/'))
  # print(testList)
  for i in (range(0,testingSetSize)):
          file = open('/user_dir' + testList[i][3:], "r")
          # file = gcs.open(test_file + testList[i])
          x = file.readline()
          # print(testList[i])
          testLabel.append(bool(int(x)))
          tempString = file.readline()
          tempArray = []
          p = 0
          for g in tempString:
              tempArray.append(int(tempString[p]))
              p+=1
          testData.append(tempArray)

          file.close()

  for i in (range(0,trainingSetSize)):
          file = open('/user_dir' + trainList[i][3:], "r")
          # file = gcs.open(train_file + trainList[i])
          x = file.readline()
          # print(trainList[i])
          trainLabel.append(bool(int(x)))
          tempString = file.readline()
          tempArray = []
          p = 0
          for g in tempString:
              tempArray.append(int(tempString[p]))
              p+=1
          trainData.append(tempArray)

          file.close()
  print("Loaded")

  cattedTest=[]
  cattedTrain=[]
  for i in range(0, trainingSetSize):
      cattedTrain.append(keras.utils.to_categorical(trainData[i]))
  for i in range(0, testingSetSize):
      cattedTest.append(keras.utils.to_categorical(testData[i]))



  x_train = np.array(cattedTrain)
  # x_train = np.asarray(cattedTrain, dtype = bool)
  y_train = np.array(trainLabel)
  x_test = np.array(cattedTest)
  # x_test = np.asarray(cattedTest, dtype = bool)
  y_test = np.array(testLabel)

  print(y_train)
  print()
  print(y_test)
  print()

  # y_trainBin = keras.utils.to_categorical(y_train)
  # y_testBin = keras.utils.to_categorical(y_test)

  # y_trainBin = np.delete(keras.utils.to_categorical(y_train), np.s_[::2], 0)
  # y_testBin = np.delete(keras.utils.to_categorical(y_test), np.s_[::2], 0)

  # print(x_train)
  # print()
  # print(y_trainBin)
  # print()
  # print(x_test)
  # print()
  # print(y_testBin)
  # print()

  # y_trainBin.reshape(1, 20)
  # y_testBin.reshape(1, 20)

  # print(x_train)
  # print()
  # print(y_trainBin)
  # print()
  # print(x_test)
  # print()
  # print(y_testBin)
  # print()
  #
  print("Moved to Numpy Array")
  x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                          value=0,
                                                          padding='post',
                                                          maxlen=15000000)

  x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                          value=0,
                                                          padding='post',
                                                          maxlen=15000000)
  # x_test=x_test.reshape(40, 2500000, 6,1)
  # x_train=x_train.reshape(40, 2500000, 6,1)
  print(x_train.shape)
  print(x_test.shape)
  print("Added Padding")

  return (x_train, y_train, x_test, y_test)

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
