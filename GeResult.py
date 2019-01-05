import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, accuracy_score
#from H5Dataset import H5Dataset
from basenet.NetworkFactory import NetworkFactory
import os
import h5py
import csv
import numpy as np

def ArrayToInputTensor(Sen1_Array, Sen2_Array):

    Input_sen1 = torch.from_numpy(Sen1_Array).permute(2,0,1).type(torch.FloatTensor)
    Input_sen1 = Input_sen1.unsqueeze(0)
    Input_sen1 = Input_sen1.cuda()

    Input_sen2 = torch.from_numpy(Sen2_Array).permute(2,0,1).type(torch.FloatTensor)
    Input_sen2 = Input_sen2.unsqueeze(0)
    Input_sen2 = Input_sen2.cuda()

    TTA_sen1, TTA_sen2 = Sen1_Array[ :, :: -1, :] - np.zeros_like(Sen1_Array), Sen2_Array[ :, :: -1, :] - np.zeros_like(Sen2_Array)
    TTA_sen1, TTA_sen2 = TTA_sen1[ :: -1, :, :] - np.zeros_like(Sen1_Array), TTA_sen2[ :: -1, :, :] - np.zeros_like(Sen2_Array)

    TTA_sen1 = torch.from_numpy(TTA_sen1).permute(2,0,1).type(torch.FloatTensor)
    TTA_sen1 = TTA_sen1.unsqueeze(0)
    TTA_sen1 = TTA_sen1.cuda()

    TTA_sen2 = torch.from_numpy(TTA_sen2).permute(2,0,1).type(torch.FloatTensor)
    TTA_sen2 = TTA_sen2.unsqueeze(0)
    TTA_sen2 = TTA_sen2.cuda()

    return Input_sen1, Input_sen2, TTA_sen1, TTA_sen2

def ArrayToLRFlipTensor(Sen1_Array, Sen2_Array):
    # flip the array right and left
    TTA_sen1, TTA_sen2 = Sen1_Array[ :, :: -1, :] - np.zeros_like(Sen1_Array), Sen2_Array[ :, :: -1, :] - np.zeros_like(Sen2_Array)

    TTA_sen1 = torch.from_numpy(TTA_sen1).permute(2,0,1).type(torch.FloatTensor)
    TTA_sen1 = TTA_sen1.unsqueeze(0)
    TTA_sen1 = TTA_sen1.cuda()

    TTA_sen2 = torch.from_numpy(TTA_sen2).permute(2,0,1).type(torch.FloatTensor)
    TTA_sen2 = TTA_sen2.unsqueeze(0)
    TTA_sen2 = TTA_sen2.cuda()

    return TTA_sen1, TTA_sen2

def ArrayToUDlipTensor(Sen1_Array, Sen2_Array):
    # flip the array up and down
    TTA_sen1, TTA_sen2 = Sen1_Array[ :: -1, :, :] - np.zeros_like(Sen1_Array), Sen2_Array[ :: -1, :, :] - np.zeros_like(Sen2_Array)

    TTA_sen1 = torch.from_numpy(TTA_sen1).permute(2,0,1).type(torch.FloatTensor)
    TTA_sen1 = TTA_sen1.unsqueeze(0)
    TTA_sen1 = TTA_sen1.cuda()

    TTA_sen2 = torch.from_numpy(TTA_sen2).permute(2,0,1).type(torch.FloatTensor)
    TTA_sen2 = TTA_sen2.unsqueeze(0)
    TTA_sen2 = TTA_sen2.cuda()

    return TTA_sen1, TTA_sen2

def gaussian(ins, mean, stddev) :   #should be moved to H5dataset
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise

def multiplicative_denoising(input_sen1, input_sen2) :

    mult_samples_sen1 = ss.gamma.rvs(1000, scale=0.001, size= input_sen1.size(0))
    mult_samples_sen1 = mult_samples_sen1[:,np.newaxis,np.newaxis,np.newaxis]
    input_sen1 = input_sen1 * torch.from_numpy(np.tile(mult_samples_sen1, [1, 8, 32, 32])).float().cuda()

    mult_samples_sen2 = ss.gamma.rvs(1000, scale=0.001, size= input_sen2.size(0))
    mult_samples_sen2 = mult_samples_sen2[:,np.newaxis,np.newaxis,np.newaxis]
    input_sen2 = input_sen2 * torch.from_numpy(np.tile(mult_samples_sen2, [1, 10, 32, 32])).float().cuda()
    return input_sen1, input_sen2

def add_noise(input1_var, input2_var):
    input1_var = gaussian(input1_var, 0, 0.01)
    input2_var = gaussian(input2_var, 0, 0.01)
    input1_var, input2_var = multiplicative_denoising(input1_var, input2_var)
    return input1_var, input2_var

def GeResult():
    # Priors

    # Dataset
    fid = h5py.File('data/round1_test_a_20181109.h5')
    Sen1_dataset = fid['sen1']
    Sen2_dataset = fid['sen2']
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    aboveThresh_index = np.load('aboveThresh_index.npy')
    anno = np.load('aboveThresh_prediction.npy')

    net_1 = NetworkFactory.ConsturctNetwork('SimpleNet' , 'weights/CEL_MIX_ValBalanceResam_bs8_CosineShedual_SimpleNet/LCZ42_SGD_11_7213.pth')
    net_1.eval()

    net_2 = NetworkFactory.ConsturctNetwork('ShallowResNeXt' , 'weights/CEL_WarmUp_TrainALlValAllBallance_bs32_8cat10channel_ShallowResNeXt/LCZ42_SGD_7_1803.pth')
    net_2.eval()

    net_3 = NetworkFactory.ConsturctNetwork('SimpleNetGN' , 'weights/FL_WarmUp_TrainALlValAllBallance_bs32_8cat10channel_Lr1e-4_SimpleNetGN/LCZ42_SGD_9_1803.pth')
    net_3.eval()

    net_4 = NetworkFactory.ConsturctNetwork('SimpleNetGN' , 'weights/CEL_WarmUp_TrainALlValAllBallance_bs32_8cat10channel_deacysqrt04_SimpleNetGN/LCZ42_SGD_11_1803.pth')
    net_4.eval()

    net_5 = NetworkFactory.ConsturctNetwork('DenseNetSia' , 'weights/CEL_WarmUp_TrainALlValAllBallance_bs32_8cat10channel_deacysqrt05_DenseNetSia/LCZ42_SGD_11_1803.pth')
    net_5.eval()
    
    sen1mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen1.npy')
    sen1std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen1.npy')

    sen2mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen2.npy')
    sen2std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen2.npy')


    results = { 'Orignal':[],  'TTA':[],  'Merged':[]}
    with open('0103.csv', 'wb') as csvfile: 
        f = csv.writer(csvfile, delimiter=',')

        for index in range(len(Sen1_dataset)):

            sen1_Array = (Sen1_dataset[index] - sen1mean)/sen1std
            sen2_Array = (Sen2_dataset[index] - sen2mean)/sen2std

            Input_sen1, Input_sen2, TTA_sen1, TTA_sen2 = ArrayToInputTensor(sen1_Array, sen2_Array)


            ConfTensor_1_orignal = net_1.forward(Input_sen1, Input_sen2)
            ConfTensor_1_TTA = net_1.forward(TTA_sen1, TTA_sen2)
            ConfTensor_1_merged = torch.nn.functional.normalize(ConfTensor_1_orignal) +torch.nn.functional.normalize(ConfTensor_1_TTA)

            ConfTensor_2_orignal = net_2.forward(Input_sen1, Input_sen2)
            ConfTensor_2_TTA = net_2.forward(TTA_sen1, TTA_sen2)

            ConfTensor_2_merged = (torch.nn.functional.normalize(ConfTensor_2_orignal) +torch.nn.functional.normalize(ConfTensor_2_TTA))/2

            ConfTensor_3_orignal = net_3.forward(Input_sen1, Input_sen2)
            ConfTensor_3_TTA = net_3.forward(TTA_sen1, TTA_sen2)

            ConfTensor_3_merged = (torch.nn.functional.normalize(ConfTensor_3_orignal) +torch.nn.functional.normalize(ConfTensor_3_TTA))/2

            ConfTensor_4_orignal = net_4.forward(Input_sen1, Input_sen2)
            ConfTensor_4_TTA = net_4.forward(TTA_sen1, TTA_sen2)

            ConfTensor_4_merged = (torch.nn.functional.normalize(ConfTensor_4_orignal) +torch.nn.functional.normalize(ConfTensor_4_TTA))/2

            ConfTensor_5_orignal = net_5.forward(Input_sen1, Input_sen2)
            ConfTensor_5_TTA = net_5.forward(TTA_sen1, TTA_sen2)

            ConfTensor_5_merged = (torch.nn.functional.normalize(ConfTensor_5_orignal) +torch.nn.functional.normalize(ConfTensor_5_TTA))/2


            ConfTensor_total = torch.nn.functional.normalize(ConfTensor_1_orignal) + ConfTensor_2_merged + ConfTensor_3_merged + ConfTensor_4_merged + ConfTensor_5_merged
            _, pred = ConfTensor_total.data.topk(1, 1, True, True)

            csvrow = []
            for i in range(17):
                if i == pred.item():
                    csvrow.append('1')
                else:
                    csvrow.append('0')
            f.writerow(csvrow)

if __name__ == '__main__':
    GeResult()
        
