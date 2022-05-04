import numpy as np
import pandas as pd
import sys


#Converts .npy file to .dat file

np_filename = sys.argv[1]
arr = np.load(np_filename, allow_pickle=False)



"""
This step is important
Numpy array is of the format given in the example
Example
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]], dtype=float32)
If we save this file then "dtype=float32" will also
get written and that will lead to failure of gsl_matrix_fread.
The solution is to call tolist() and then convert to numpy again.
This way, the new array becomes
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
And this will be easily read by gsl_matrix_fread.
"""
arr_temp = np.array(arr.tolist())



print(arr.shape)
dot_index = np_filename.index('.')
dat_filename = np_filename[:dot_index] + ".dat"
fileobj = open(dat_filename, mode='wb')

arr_temp.tofile(fileobj)
fileobj.close()