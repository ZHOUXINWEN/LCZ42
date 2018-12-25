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
from basenet.SimpleNet import SimpleNet, SimpleNetLeaky
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
    mode = 'round1_test_a_20181109'
    '''
    826
    filepath_1 = 'weights/MIX_CEL_lr4e-3_bs8_8cat10channel_SimpleNet/LCZ42_SGD_7_1999'
    filepath_2 = 'weights/ReMIX_CEL_lr4e-3_bs8_8cat10channel_SimpleNet/LCZ42_SGD_3_1999'
    '''
    filepath_1 = 'weights/CEL_Tiers23_bs8_8cat10channel_SimpleNet/LCZ42_SGD_4_11999'
    filepath_2 = 'weights/CEL_Tiers13_bs8_8cat10channel_SimpleNet/LCZ42_SGD_4_15999'
    # Dataset
    fid = h5py.File('data/' + mode + '.h5')
    Sen1_dataset = fid['sen1']
    Sen2_dataset = fid['sen2']
    #Target_tensor = fid['label']
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    #filepath_2 = 'weights/likeval826_CEL_bs8_8cat10channel_SimpleNet/LCZ42_SGD_9_1999'

    # load Network 1
    Network_1 = SimpleNet(17)
    net_1 = torch.nn.DataParallel(Network_1, device_ids=[0])
    cudnn.benchmark = True
    Network_1.load_state_dict(torch.load(filepath_1 + '.pth'))
    net_1.eval()
    # load Network 2
    Network_2 = SimpleNet(17)
    net_2 = torch.nn.DataParallel(Network_2, device_ids=[0])
    cudnn.benchmark = True
    Network_2.load_state_dict(torch.load(filepath_2 +'.pth'))
    net_2.eval()
    
    # initialisation the random forest and load the weight
    #clf = RandomForestClassifier(n_estimators = 40, max_features = 'log2')
    #clf = joblib.load('100_estimator_max_features_log2_RandomForest.pkl')

    #weight for each model
    weight_1 = np.load(filepath_1 + '.npy')
    weight_1 = weight_1.astype('float') / weight_1.sum(axis=0)[np.newaxis, :]    #calculate precision
    #weight_1 = weight_1.astype('float') / weight_1.sum(axis=1)[:, np.newaxis]     #calculate recall
    PreciWeight_1 = torch.diag(torch.from_numpy(weight_1))

    weight_2 = np.load(filepath_2 + '.npy')
    weight_2 = weight_2.astype('float') / weight_2.sum(axis=0)[np.newaxis, :]    #calculate precision
    #weight_2 = weight_2.astype('float') / weight_2.sum(axis=1)[:, np.newaxis]     #calculate recall 
    PreciWeight_2 = torch.diag(torch.from_numpy(weight_2))

    concated_weight = torch.cat((PreciWeight_1.view(17,-1), PreciWeight_2.view(17,-1)), 1)
    PW = torch.nn.functional.normalize(concated_weight, 1)   #concat two weight and normalise them
    PW_1 = PreciWeight_1.gt(PreciWeight_2)
    PW_2 = PreciWeight_2.gt(PreciWeight_1)

    # load mean and std
    sen1mean = np.load('data/mean_et_std/' + mode + '_mean_sen1.npy')
    sen1std = np.load('data/mean_et_std/' + mode + '_std_sen1.npy')

    sen2mean = np.load('data/mean_et_std/' + mode + '_mean_sen2.npy')
    sen2std = np.load('data/mean_et_std/' + mode + '_std_sen2.npy')
    indices = []

    #open csv file and write result
    with open('MergeTiers_23_13.csv', 'wb') as csvfile: # with open as   closed automatically
        f = csv.writer(csvfile, delimiter=',')

        for index in range(len(Sen1_dataset)):
            Input_sen1 = (Sen1_dataset[index] - sen1mean)/sen1std
            Input_sen2 = (Sen2_dataset[index] - sen2mean)/sen2std

            #AnnoTensor = torch.from_numpy(Target_tensor[index])
            #Anno = torch.squeeze(torch.nonzero(AnnoTensor)).item()

            sen2nonstd = torch.from_numpy(Input_sen2*sen2std).permute(2,0,1).type(torch.FloatTensor)
            sen2nonstd = sen2nonstd.unsqueeze(0)

            
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


            preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_2)*PW[:, 1].float().cuda()
            conf, pred = preds.data.topk(1, 1, True, True)
            
            #RF_Pred = clf.predict(preds[0].detach().cpu().numpy().reshape(1,-1)).tolist()
            csvrow = []
            '''
            if (pred.item() == Anno):
                indices.append(index)
            '''
            if(index%1000 == 0) :
                print(index)
            for i in range(17):
                if i == pred.item():
                    csvrow.append('1')
                else:
                    csvrow.append('0')
            f.writerow(csvrow)
    #np.save('indices_that_like_val_V2_826.npy', indices)
if __name__ == '__main__':
    GeResult()
        
