'''
This fiel is to resample the dataset to be balance between class
'''

import h5py
import torch
import random
import numpy as np

indices = []

indicesPerClass = np.load('/home/zxw/LCZ42/data/validation_IndexEachClass.npy')
No_of_each_class = np.load('/home/zxw/LCZ42/data/number_of_class_inval.npy')
MAX = float(np.max(No_of_each_class))
mutilplier = []

for i in range(17):
    mutilplier.append(MAX/No_of_each_class[i])

"""
mutilplier = {0:9.73934491,  1:2.02034301,  2:1.55741015,  3:5.70558317,  4:2.99272419,
        5:1.39866818, 6:15.09911288,  7:1.25512384,  8:3.63361307,  9:4.12907813,
        10:1.1505058 ,  11:5.18803868,  12:5.38559738,  13:1.1929091 , 14:20.63503344,
        15:6.24955685}
"""
indices = []
for i in range(17):
    integer = int(mutilplier[i])
    fracNum = int( (mutilplier[i] - integer)*len(indicesPerClass[i]) )
    print(mutilplier[i], fracNum, len(indicesPerClass[i]))
    if fracNum != 0 :
        indices = indices + indicesPerClass[i]*integer + random.sample(indicesPerClass[i], fracNum)
    else :
        indices = indices + indicesPerClass[i]*integer
print(len(indices))
np.save('MulResampleIndex_forval.npy', indices)

