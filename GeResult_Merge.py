"""
This file is to merge the result of multi models (by weight of precison(or recall)) (TBC: by Voting)
"""
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from basenet.pnasnet import pnasnet5large
from basenet.resnext import CifarResNeXt, ShallowResNeXt
from basenet.densenet import DenseNet
from basenet.senet_shallow import se_resnet50_shallow
from basenet.SimpleNet import SimpleNet, SimpleNetLeaky,SimpleNetGN
from basenet.senet import se_resnext101_32x4d,se_resnet101,se_resnet50
import os
import h5py
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def ConstructNetwork(filepath, NetworkType, returnweight = True):
    if NetworkType == 'SimpleNet' :
        Network = SimpleNet(17)
    elif NetworkType == 'SimpleNetGN' :
        Network = SimpleNetGN(17)
    elif NetworkType == 'DenseNet' :
        Network = DenseNet(num_classes = 17)
    elif NetworkType == 'ShallowResNeXt':
        Network = ShallowResNeXt(num_classes = 17, depth = 11, cardinality = 16)
    net = torch.nn.DataParallel(Network, device_ids=[0])
    cudnn.benchmark = True
    Network.load_state_dict(torch.load(filepath + '.pth'))
    net.eval()
    if returnweight :
        weight = np.load(filepath + '.npy')
        #weight_1 = weight_1.astype('float') / weight_1.sum(axis=0)[np.newaxis, :]    #calculate precision
        weight = weight.astype('float') / weight.sum(axis=1)[:, np.newaxis]     #calculate recall
        PreciWeight = torch.diag(torch.from_numpy(weight))
        return net, PreciWeight
    else :
        return net


def GeResultFromNN():
    # Priors
    cnt = 0
    # Dataset
    fid = h5py.File('data/round1_test_a_20181109.h5')
    Sen1_dataset = fid['sen1']
    Sen2_dataset = fid['sen2']
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    # Network
    filepath_1 = 'weights/CEL_MIX_ValBalanceResam_bs8_CosineShedual_SimpleNet/LCZ42_SGD_11_7213'
    filepath_2 = 'weights/CEL_WarmUp_TrainALlValAllBallance_bs32_8cat10channel_ShallowResNeXt/LCZ42_SGD_7_1803'
    filepath_3 = 'weights/CEL_Tiers12_bs8_8cat10channel_SimpleNet/LCZ42_SGD_11_2999'
    filepath_4 = 'weights/CEL_symme_Tiers23_bs32_8cat10channel_SimpleNetGN/LCZ42_SGD_15_1803'

    # construct network
    net_1, PreciWeight_1 = ConstructNetwork(filepath_1, 'SimpleNet')
    net_2 = ConstructNetwork(filepath_2, 'ShallowResNeXt', returnweight = False)
    net_3, PreciWeight_3 = ConstructNetwork(filepath_3, 'SimpleNet')
    net_4, PreciWeight_4 = ConstructNetwork(filepath_4, 'SimpleNetGN')

    #concat  weights and normalise them
    #PW = torch.nn.functional.normalize(torch.cat((PreciWeight_1.view(17,-1), PreciWeight_2.view(17,-1), PreciWeight_3.view(17,-1)), 1), 1)   

    # load mean and std
    sen1mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen1.npy')
    sen1std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen1.npy')

    sen2mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen2.npy')
    sen2std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen2.npy')

    #open csv file and write result
    number = 0
    with open('1226.csv', 'wb') as csvfile: # with open as   closed automatically
        f = csv.writer(csvfile, delimiter=',')

        for index in range(len(Sen1_dataset)):
            Input_sen1 = (Sen1_dataset[index] - sen1mean)/sen1std
            Input_sen2 = (Sen2_dataset[index] - sen2mean)/sen2std
            
            Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
            Input_sen1 = Input_sen1.unsqueeze(0)

            Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
            Input_sen2 = Input_sen2.unsqueeze(0)
 
            preds_1 = net_1.forward(Input_sen1.cuda(), Input_sen2.cuda())
            preds_2 = net_2.forward(Input_sen1.cuda(), Input_sen2.cuda())
            #preds_3 = net_3.forward(Input_sen1.cuda(), Input_sen2.cuda())
            #preds_4 = net_4.forward(Input_sen1.cuda(), Input_sen2.cuda())

            preds = torch.nn.functional.normalize(preds_1) + torch.nn.functional.normalize(preds_2) #+ torch.nn.functional.normalize(preds_3) + torch.nn.functional.normalize(preds_4)
            #preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_2)*PW[:, 1].float().cuda() + torch.nn.functional.normalize(preds_3)*PW[:, 2].float().cuda()
            conf, pred = preds.data.topk(1, 1, True, True)      # the first bool parameter is to decide Top1(True) or Last1(False)
            # to get calss that has the minimum confidence, use topk(1, 1, False, False)
            '''                        
            if conf.item() > 0.55 :         #if >0.55   3520 0.728
                number = number + 1
            '''
            csvrow = []
            for i in range(17):
                if i == pred.item():
                    csvrow.append('1')
                else:
                    csvrow.append('0')
            f.writerow(csvrow)
    #print(number)
if __name__ == '__main__':
    GeResultFromNN()
        
