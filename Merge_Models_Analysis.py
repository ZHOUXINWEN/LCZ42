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
from basenet.SimpleNet import SimpleNet, SimpleNetLeaky, SimpleNet4x4, SimpleNetGN
from basenet.pnasnet import pnasnet5large
from basenet.senet_shallow import se_resnet50_shallow
from basenet.senet import se_resnext101_32x4d,se_resnet101,se_resnet50

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  #recall
        #cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

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

def GeResult():
    # Dataset
    Dataset_validation = H5DatasetSia(root = 'data', mode = 'validation',symmetrize = False)
    Dataloader_validation = data.DataLoader(Dataset_validation, batch_size = 1,
                                 num_workers = 1,
                                 shuffle = True, pin_memory = True)
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    class_names = ['Compact high-rise', 'Compact midrise', 'Compact lowrise','Open high-rise', 'Open midrise', 'Open lowrise', 'Lightweight low-rise', 'Large low-rise', 'Sparsely built',
               'Heavy industry', 'Dense trees', 'Scattered trees', 'Bush and scrub', 'Low plants', 'Bare rock or paved', 'Bare soil or sand', 'Water']
    
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

    # initialisation the random forest and load the weight
    clf = RandomForestClassifier(n_estimators = 100, max_features = 'log2')
    clf = joblib.load('100_estimator_max_features_log2_RandomForest.pkl')

    #PW = torch.nn.functional.normalize(torch.cat((PreciWeight_1.view(17,-1), PreciWeight_3.view(17,-1)), 1), 1)
    #PW = torch.nn.functional.normalize(torch.cat((PreciWeight_1.view(17,-1), PreciWeight_2.view(17,-1), PreciWeight_3.view(17,-1)), 1), 1)   #concat two weight and normalise them
    result_tensor = []
    result_permodel = [[],[],[],[],[]]
    results = { 'Pred':[],  'Conf':[],  'Anno':[]}
    results_underThresh = { 'Pred':[],  'Anno':[]}
    results_anno = []
    for i, (Input_sen1, Input_sen2, Anno) in enumerate(Dataloader_validation):
        Input_sen1 = Input_sen1.cuda()
        Input_sen2 = Input_sen2.cuda()

        preds_1 = net_1.forward(Input_sen1, Input_sen2)
        conf_1, pred_1 = preds_1.data.topk(1, 1, True, True)
        result_permodel[1].append(pred_1.item())


        preds_2 = net_2.forward(Input_sen1, Input_sen2)
        conf_2, pred_2 = preds_2.data.topk(1, 1, True, True)
        result_permodel[2].append(pred_2.item())


        preds_3 = net_3.forward(Input_sen1, Input_sen2)
        conf_3, pred_3 = preds_3.data.topk(1, 1, True, True)
        result_permodel[3].append(pred_3.item())


        preds_4 = net_4.forward(Input_sen1, Input_sen2)
        conf_4, pred_4 = preds_4.data.topk(1, 1, True, True)
        result_permodel[4].append(pred_3.item())


        '''
        features = Network_3.features(Input_sen1.cuda(), Input_sen2.cuda())
        features256 = Network_3.features_256(features)
        RF_Pred = clf.predict(features256[0].detach().cpu().numpy().reshape(1,-1)).tolist()
        result_permodel[4].append(RF_Pred[0])
        #print(RF_Pred[0], Anno)
        '''
        preds = torch.nn.functional.normalize(preds_1) + torch.nn.functional.normalize(preds_2) #+ torch.nn.functional.normalize(preds_3) + torch.nn.functional.normalize(preds_4)
        #preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_3)*PW[:, 1].float().cuda()
        #preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_2)*PW[:, 1].float().cuda() + torch.nn.functional.normalize(preds_3)*PW[:, 2].float().cuda()
        #RF_Pred = clf.predict(features256[0].detach().cpu().numpy().reshape(1,-1)).tolist()

        result_tensor.append(preds[0].detach().cpu().numpy().tolist())
        #preds = -preds
        conf, pred = preds.data.topk(1, 1, True, True)

        results_anno.append(Anno)
        #append prediction results
        results['Pred'].append(pred.item())#RF_Pred[0])
        results['Conf'].append(conf.item())
        results['Anno'].append(Anno)
       
        if(i%10000 == 0):
            print(i)
        #append annotation results

        if conf > 0.85:
            results_underThresh['Pred'].append(pred.item())
            #append annotation results
            results_underThresh['Anno'].append(Anno.item())

    
    #accuracy of each model
    print('Accuracy of model 1: %0.6f'%(accuracy_score(result_permodel[1], results['Anno'], normalize = True)))
    print('Accuracy of model 2: %0.6f'%(accuracy_score(result_permodel[2], results['Anno'], normalize = True)))
    print('Accuracy of model 3: %0.6f'%(accuracy_score(result_permodel[3], results['Anno'], normalize = True)))
    print('Accuracy of model 4: %0.6f'%(accuracy_score(result_permodel[4], results['Anno'], normalize = True)))
    #print('Accuracy of model RF: %0.6f'%(accuracy_score(result_permodel[4], results['Anno'], normalize = True)))
    #similarity between each model
    print('Similarty model 1 and 2: %d'%(accuracy_score(result_permodel[1], result_permodel[2], normalize = False)))
    print('Similarty model 1 and 3: %d'%(accuracy_score(result_permodel[1], result_permodel[3], normalize = False)))
    print('Similarty model 2 and 3: %d'%(accuracy_score(result_permodel[2], result_permodel[3], normalize = False)))
    print(len(results_underThresh['Pred']), len(results_underThresh['Pred'])/len(Dataloader_validation)) 

    print('Accuracy under threshold: %0.6f'%(accuracy_score(results_underThresh['Pred'], results_underThresh['Anno'], normalize = True)))

    print('Accuracy of merged Models: %0.6f'%(accuracy_score(results['Pred'], results['Anno'], normalize = True)))
    cnf_matrix = confusion_matrix(results['Anno'], results['Pred'])
    np.save('results/confmat_Merge.npy',cnf_matrix)

    # Plot non-normalized confusion matrix
    cnf_tr = np.trace(cnf_matrix)
    cnf_tr = cnf_tr.astype('float')

    print(cnf_tr/len(Dataset_validation))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names ,title='Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
if __name__ == '__main__':
    GeResult()
        
