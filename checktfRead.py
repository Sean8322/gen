import os
import subprocess

WORKING_DIR = os.getcwd()

subprocess.check_call(
  ['gsutil', 'cp', 'gs://oxygen-bac/bio/train/CP001858.txt', raw_local_files_data_paths[i]])
