import torch
import h5py
import numpy as np
import time
from loss.DistillationLoss import DistillationLoss
from loss.SmoothingLoss import SmoothingLoss
import scipy.stats as ss
import scipy.misc as sm


fid = h5py.File('data/round1_test_a_20181109.h5')
Sen1_dataset = fid['sen1']
Sen2_dataset = fid['sen2']
torch.set_default_tensor_type('torch.cuda.FloatTensor')



