"""
Analysis performance of Model ensembles by testing them in validation set
"""
import os
import csv
import h5py
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, accuracy_score
from H5Dataset import H5DatasetSia
from basenet.resnext import CifarResNeXt, ShallowResNeXt
from basenet.SimpleNet import SimpleNet, SimpleNetLeaky, SimpleNet4x4
from basenet.pnasnet import pnasnet5large
from basenet.senet_shallow import se_resnet50_shallow
from basenet.senet import se_resnext101_32x4d,se_resnet101,se_resnet50

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def GeResult():
    # Priors
    test = False
    # Dataset
    mode = 'training'
    fid = h5py.File('data/' + mode + '.h5')
    Sen1_dataset = fid['sen1']
    Sen2_dataset = fid['sen2']

    if test != True :
        label = fid['label']

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    class_names = ['Compact high-rise', 'Compact midrise', 'Compact lowrise','Open high-rise', 'Open midrise', 'Open lowrise', 'Lightweight low-rise', 'Large low-rise', 'Sparsely built',
               'Heavy industry', 'Dense trees', 'Scattered trees', 'Bush and scrub', 'Low plants', 'Bare rock or paved', 'Bare soil or sand', 'Water']
    
    # Network
    filepath_1 = 'weights/CEL_MIX_ValBalanceResam_bs8_CosineShedual_SimpleNet/LCZ42_SGD_11_7213'
    filepath_2 = 'weights/CEL_Tiers13_bs8_8cat10channel_SimpleNet/LCZ42_SGD_11_1999'
    filepath_3 = 'weights/CEL_Tiers12_bs8_8cat10channel_SimpleNet/LCZ42_SGD_11_2999'
    #Network_1 = ShallowResNeXt(num_classes = 17, depth = 11, cardinality = 16)
    Network_1 = SimpleNet(17)
    net_1 = torch.nn.DataParallel(Network_1, device_ids=[0])
    cudnn.benchmark = True    
    Network_1.load_state_dict(torch.load(filepath_1 + '.pth'))
    net_1.eval()


    Network_2 = SimpleNet(17)
    net_2 = torch.nn.DataParallel(Network_2, device_ids=[0])
    cudnn.benchmark = True
    Network_2.load_state_dict(torch.load(filepath_2 + '.pth'))
    net_2.eval()

    Network_3 = SimpleNet(17)       #Network_3 = se_resnet50_shallow(17, None)
    net_3 = torch.nn.DataParallel(Network_3, device_ids=[0])
    cudnn.benchmark = True
    Network_3.load_state_dict(torch.load(filepath_3 + '.pth'))
    Network_3.eval()
    net_3.eval()

    # initialisation the random forest and load the weight
    clf = RandomForestClassifier(n_estimators = 100, max_features = 'log2')
    clf = joblib.load('100_estimator_max_features_log2_RandomForest.pkl')

    #load weight determined by precision
    weight_1 = np.load(filepath_1 + '.npy')
    #weight_1 = weight_1.astype('float') / weight_1.sum(axis=0)[np.newaxis, :]    #calculate precision
    weight_1 = weight_1.astype('float') / weight_1.sum(axis=1)[:, np.newaxis]     #calculate recall
    PreciWeight_1 = torch.diag(torch.from_numpy(weight_1))

    weight_2 = np.load(filepath_2 + '.npy')
    #weight_2 = weight_2.astype('float') / weight_2.sum(axis=0)[np.newaxis, :]    #calculate precision
    weight_2 = weight_2.astype('float') / weight_2.sum(axis=1)[:, np.newaxis]     #calculate recall 
    PreciWeight_2 = torch.diag(torch.from_numpy(weight_2))

    weight_3 = np.load(filepath_3 + '.npy')
    #weight_3 = weight_3.astype('float') / weight_3.sum(axis=0)[np.newaxis, :]    #calculate precision
    weight_3 = weight_3.astype('float') / weight_3.sum(axis=1)[:, np.newaxis]     #calculate recall 
    PreciWeight_3 = torch.diag(torch.from_numpy(weight_3))

    #PW = torch.nn.functional.normalize(torch.cat((PreciWeight_1.view(17,-1), PreciWeight_3.view(17,-1)), 1), 1)
    PW = torch.nn.functional.normalize(torch.cat((PreciWeight_1.view(17,-1), PreciWeight_2.view(17,-1), PreciWeight_3.view(17,-1)), 1), 1)  
    #concat two weight and normalise them

    sen1mean = np.load('data/mean_et_std/' + mode + '_mean_sen1.npy')
    sen1std = np.load('data/mean_et_std/' + mode + '_std_sen1.npy')

    sen2mean = np.load('data/mean_et_std/' + mode + '_mean_sen2.npy')
    sen2std = np.load('data/mean_et_std/' + mode + '_std_sen2.npy')

    correct_index = []
    result_tensor = []
    result_permodel = [[],[],[],[],[]]
    results = { 'Pred':[],  'Conf':[],  'Anno':[]}
    results_underThresh = { 'Pred':[],  'Anno':[]}
    results_anno = []
    soft_labels = []
    for index in range(len(Sen1_dataset)):

        Input_sen1 = (Sen1_dataset[index] - sen1mean)/sen1std
        Input_sen2 = (Sen2_dataset[index] - sen2mean)/sen2std

        Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
        Input_sen1 = Input_sen1.unsqueeze(0)

        Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
        Input_sen2 = Input_sen2.unsqueeze(0)

        Input_sen1 = Input_sen1.cuda()
        Input_sen2 = Input_sen2.cuda()

        if test != True :
            AnnoTensor = torch.from_numpy(label[index])
            Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()
        #sen2nonstd = sen2nonstd.cuda()

        preds_1 = net_1.forward(Input_sen1, Input_sen2)
        #preds_1normal = torch.nn.functional.normalize(preds_1)
        conf_1, pred_1 = preds_1.data.topk(1, 1, True, True)
        result_permodel[1].append(pred_1.item())


        preds_2 = net_2.forward(Input_sen1, Input_sen2)
        #preds_2normal = torch.nn.functional.normalize(preds_2)
        conf_2, pred_2 = preds_2.data.topk(1, 1, True, True)
        result_permodel[2].append(pred_2.item())


        preds_3 = net_3.forward(Input_sen1, Input_sen2)
        #preds_3normal = torch.nn.functional.normalize(preds_3)
        conf_3, pred_3 = preds_3.data.topk(1, 1, True, True)
        result_permodel[3].append(pred_3.item())


        #preds = preds_1 + preds_2
        #preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_3)*PW[:, 1].float().cuda()
        preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_2)*PW[:, 1].float().cuda() + torch.nn.functional.normalize(preds_3)*PW[:, 2].float().cuda()

        conf, pred = preds.data.topk(1, 1, True, True)
        #if(pred.item() == np.nonzero(label[index])[0][0]):
            #print(pred.item(), np.nonzero(label[index])[0][0], index)
        soft_labels.append(preds[0].detach().cpu().numpy().tolist())
        #correct_index.append(index)
        #results_anno.append(Anno)
        #append prediction results
        results['Pred'].append(pred.item())#RF_Pred[0])
        results['Conf'].append(conf.item())
        results['Anno'].append(np.nonzero(label[index])[0][0])
     
        if(index%1000 == 0):
            print(index)
        #append annotation results
        #print(conf.item())
    
    np.save(mode + '_soft_labels_851.npy',soft_labels)
    #np.save(mode + '_correct_index_851.npy',correct_index)
    #np.save('round1_test_a_20181109_results_anno.npy', results['Pred'])
    #np.save('results.npy', results)
    print('Accuracy of merged Models: %0.6f'%(accuracy_score(results['Pred'], results['Anno'], normalize = True)))

if __name__ == '__main__':
    GeResult()
        
