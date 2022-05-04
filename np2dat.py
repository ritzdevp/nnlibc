import numpy as np
import pandas as pd
import sys


#Converts .npy file to .dat file

np_filename = sys.argv[1]
arr = np.load(np_filename)

dot_index = np_filename.index('.')
dat_filename = np_filename[:dot_index] + ".dat"
fileobj = open(dat_filename, mode='wb')

arr.tofile(fileobj)
fileobj.close()