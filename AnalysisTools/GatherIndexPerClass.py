'''
Put the index of the samples that has same class in a list

'''

import h5py
import torch
import numpy as np

mode = 'training'
filename = 'data/' + mode + '.h5'
File = h5py.File(filename, 'r')
labels = File['label']
lists = [[] for i in range(17)]
for index in range(len(labels)):
    AnnoTensor = torch.from_numpy(labels[index])
    Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
    lists[Anno].append(index)

np.save('data/' + mode + '_IndexEachClass.npy', lists)

