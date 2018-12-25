'''
find out the index that is not sampled
'''
import h5py
import torch
import numpy as np

LikeValIndice = np.load('rand_half_1.npy')
mode = 'training'
filename = '/home/zxw/LCZ42/data/' + mode + '.h5'
File = h5py.File(filename, 'r')
labels = File['label']
"""
lists = [[] for i in range(17)]
weights = []
for idx in range(len(LikeValIndice)):
    index = LikeValIndice[idx]
    AnnoTensor = torch.from_numpy(labels[index])
    Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
    lists[Anno].append(index)
for i in range(17):
    print(i, len(lists[i]))
    
np.save(mode + '_IndexEachClass_likeVal.npy', lists)
"""
NotFitbyRF = []
j = 0
for i in range(len(labels)):
    if(LikeValIndice[j] != i):
        NotFitbyRF.append(i)
        i = i + 1
    else:
        j = j + 1
print(len(NotFitbyRF))
np.save('NotFitByRF.npy', NotFitbyRF)
