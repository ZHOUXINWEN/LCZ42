'''
This File is used to train Random Forest from 256 features of Single Model

'''
import os
import csv
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from basenet.senet import se_resnext101_32x4d,se_resnet101,se_resnet50
from basenet.SimpleNet import SimpleNet, SimpleNetLeaky, SimpleNet4x4
from basenet.senet_shallow import se_resnet50_shallow
from H5Dataset import H5Dataset, H5DatasetSia



def GeResult():
    # Priors
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Dataset
    Dataset_validation = H5DatasetSia(root = 'data', mode = 'validation')
    Dataloader_validation = data.DataLoader(Dataset_validation, batch_size = 1,
                                 num_workers = 1,
                                 shuffle = True, pin_memory = True)

    # Network
    Network = SimpleNet(17)
    #net = torch.nn.DataParallel(Network, device_ids=[0])
    cudnn.benchmark = True    
    Network.load_state_dict(torch.load('weights/CEL_Tiers12_bs8_8cat10channel_SimpleNet/LCZ42_SGD_11_2999.pth'))
    Network.eval()

    results = []
    results_anno = []

    for i, (Input_sen1, Input_sen2, Anno) in enumerate(Dataloader_validation):
        
        features = Network.features(Input_sen1.cuda(), Input_sen2.cuda())
        features256 = Network.features_256(features)

        results.append(features256[0].detach().cpu().numpy().tolist())

        results_anno.append(Anno)
        if((i+1)%1000 == 0) :
            print( i+1)

    np.save('data/features/845_features256_validation_results.npy', results)
    np.save('data/features/845_features256_training_results_anno.npy', results_anno)

if __name__ == '__main__':
    GeResult()
        
