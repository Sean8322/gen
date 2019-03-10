import sys
import os
filenames = os.listdir(sys.path[0] + "/examples/")

for i in range (0, 116):
    fin = open(sys.path[0] + '/examples/' + filenames[i], 'r', encoding = 'us-ascii')
    trash = fin.readline()
    data = fin.readline()
    fout = open(sys.path[0] + '/examples/' + filenames[i], 'w', encoding = 'us-ascii')
    fout.writelines(data)
