import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#from H5Dataset import H5Dataset
from basenet.SimpleNet import SimpleNet

import os
import h5py
import csv
import numpy as np

def GeResult():
    # Priors

    # Dataset
    fid = h5py.File('data/round1_test_a_20181109.h5')
    Sen1_dataset = fid['sen1']
    Sen2_dataset = fid['sen2']
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Network
    
    #Network = pnasnet5large(6, None)
    #Network = ResNeXt101_64x4d(6)
    Network = SimpleNet(17)                      #Network_1 = se_resnet50_shallow(17, None)
    #Network = se_resnet50(17, None)
    net = torch.nn.DataParallel(Network, device_ids=[0])
    cudnn.benchmark = True
    
    Network.load_state_dict(torch.load('weights/Distillation_MIX_ValBalanceResam_bs8_SimpleNet/LCZ42_SGD_11_7213.pth'))
    net.eval()

    sen1mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen1.npy')
    sen1std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen1.npy')

    sen2mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen2.npy')
    sen2std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen2.npy')

    results = []
    with open('Distillation_11_7213.csv', 'wb') as csvfile: # with open as   closed automatically
        f = csv.writer(csvfile, delimiter=',')

        for index in range(len(Sen1_dataset)):
            Input_sen1 = (Sen1_dataset[index] - sen1mean)/sen1std
            Input_sen2 = (Sen2_dataset[index] - sen2mean)/sen2std

            Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
            Input_sen1 = Input_sen1.unsqueeze(0)
            Input_sen1 = Input_sen1.cuda()

            Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
            Input_sen2 = Input_sen2.unsqueeze(0)
            Input_sen2 = Input_sen2.cuda()

            preds = net.forward(Input_sen1, Input_sen2)
            _, pred = preds.data.topk(1, 1, True, True)

            results.append( pred.item())
            csvrow = []
            for i in range(17):
                if i == pred.item():
                    csvrow.append('1')
                else:
                    csvrow.append('0')
            f.writerow(csvrow)
        np.save('Single_model_result.npy', results)

if __name__ == '__main__':
    GeResult()
        
