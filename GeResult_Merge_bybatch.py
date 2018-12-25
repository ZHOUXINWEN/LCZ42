"""
This file is to merge the result of multi models (by weight of precison(or recall)) (TBC: by Voting)
"""
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from basenet.pnasnet import pnasnet5large
from basenet.resnext import CifarResNeXt, ShallowResNeXt
from basenet.senet_shallow import se_resnet50_shallow
from basenet.SimpleNet import SimpleNet, SimpleNetLeaky,SimpleNetGN
from basenet.senet import se_resnext101_32x4d,se_resnet101,se_resnet50
import os
import h5py
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def GeResult():
    # Priors
    cnt = 0
    # Dataset
    fid = h5py.File('data/round1_test_a_20181109.h5')
    Sen1_dataset = fid['sen1']
    Sen2_dataset = fid['sen2']
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    filepath_1 = 'weights/CEL_MIX_ValBalanceResam_bs8_CosineShedual_SimpleNet/LCZ42_SGD_11_7213'
    filepath_2 = 'weights/CEL_Tiers13_bs8_8cat10channel_SimpleNet/LCZ42_SGD_11_1999'
    filepath_3 = 'weights/CEL_Tiers12_bs8_8cat10channel_SimpleNet/LCZ42_SGD_11_2999'
    filepath_4 = 'weights/CEL_symme_Tiers12_bs32_8cat10channel_SimpleNetGN/LCZ42_SGD_15_753'

    # load Network 1
    Network_1 = SimpleNet(17)                      #Network_1 = se_resnet50_shallow(17, None)
    net_1 = torch.nn.DataParallel(Network_1, device_ids=[0])
    cudnn.benchmark = True
    Network_1.load_state_dict(torch.load(filepath_1 + '.pth'))
    net_1.eval()

    Network_2 = SimpleNet(17)
    net_2 = torch.nn.DataParallel(Network_2, device_ids=[0])
    cudnn.benchmark = True
    Network_2.load_state_dict(torch.load(filepath_2 +'.pth'))
    net_2.eval()

    Network_3 = SimpleNet(17)      #Network_3 = se_resnet50_shallow(17, None)
    net_3 = torch.nn.DataParallel(Network_3, device_ids=[0])
    cudnn.benchmark = True
    Network_3.load_state_dict(torch.load(filepath_3 + '.pth'))
    Network_3.eval()
    net_3.eval()

    Network_4 = SimpleNetGN(17) 
    net_4 = torch.nn.DataParallel(Network_4, device_ids=[0])
    cudnn.benchmark = True
    Network_4.load_state_dict(torch.load(filepath_4 + '.pth'))
    net_4.eval()
    
    # initialisation the random forest and load the weight
    #clf = RandomForestClassifier(n_estimators = 100, max_features = 'log2')
    #clf = joblib.load('100_estimator_max_features_log2_RandomForest.pkl')

    #weight for each model
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

    PW = torch.nn.functional.normalize(torch.cat((PreciWeight_1.view(17,-1), PreciWeight_2.view(17,-1), PreciWeight_3.view(17,-1)), 1), 1)   #concat two weight and normalise them

    # load mean and std
    sen1mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen1.npy')
    sen1std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen1.npy')

    sen2mean = np.load('data/mean_et_std/round1_test_a_20181109_mean_sen2.npy')
    sen2std = np.load('data/mean_et_std/round1_test_a_20181109_std_sen2.npy')
    NN851 =[]
    #open csv file and write result
    with open('1223.csv', 'wb') as csvfile: # with open as   closed automatically
        f = csv.writer(csvfile, delimiter=',')

        for index in range(len(Sen1_dataset)):
            Input_sen1 = (Sen1_dataset[index] - sen1mean)/sen1std
            Input_sen2 = (Sen2_dataset[index] - sen2mean)/sen2std
            
            Input_sen1 = torch.from_numpy(Input_sen1).permute(2,0,1).type(torch.FloatTensor)
            Input_sen1 = Input_sen1.unsqueeze(0)


            Input_sen2 = torch.from_numpy(Input_sen2).permute(2,0,1).type(torch.FloatTensor)
            Input_sen2 = Input_sen2.unsqueeze(0)


            preds_1 = net_1.forward(Input_sen1.cuda(), Input_sen2.cuda())
            #_, pred_1 = preds_1.data.topk(1, 1, True, True)
            #result_1.append(pred_1.item())

            preds_2 = net_2.forward(Input_sen1.cuda(), Input_sen2.cuda())
            #_, pred_2 = preds_2.data.topk(1, 1, True, True)
            #result_2.append(pred_1.item())
            preds_3 = net_3.forward(Input_sen1.cuda(), Input_sen2.cuda())

            preds_4 = net_4.forward(Input_sen1.cuda(), Input_sen2.cuda())
            preds = (torch.nn.functional.normalize(preds_1) + torch.nn.functional.normalize(preds_2) + torch.nn.functional.normalize(preds_3) + torch.nn.functional.normalize(preds_4))
            #preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_2)*PW[:, 1].float().cuda() + torch.nn.functional.normalize(preds_3)*PW[:, 2].float().cuda()
            conf, pred = preds.data.topk(1, 1, True, True)
            
            #RF_Pred = clf.predict(preds[0].detach().cpu().numpy().reshape(1,-1)).tolist()  
            #class_label = 18
            csvrow = []
            #RF851.append(RF_Pred[0])
            NN851.append(pred.item())
            for i in range(17):
                if i == pred.item():
                    csvrow.append('1')
                else:
                    csvrow.append('0')
            f.writerow(csvrow)
    np.save('NN851byBacth.npy', NN851)
if __name__ == '__main__':
    GeResult()
        
