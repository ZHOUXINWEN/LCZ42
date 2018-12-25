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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

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

def GeResult():
    # Priors

    # Dataset
    Dataset_validation = H5DatasetSia(root = 'data', mode = 'validation')
    Dataloader_validation = data.DataLoader(Dataset_validation, batch_size = 1,
                                 num_workers = 1,
                                 shuffle = True, pin_memory = True)
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    class_names = ['Compact high-rise', 'Compact midrise', 'Compact lowrise','Open high-rise', 'Open midrise', 'Open lowrise', 'Lightweight low-rise', 'Large low-rise', 'Sparsely built',
               'Heavy industry', 'Dense trees', 'Scattered trees', 'Bush and scrub', 'Low plants', 'Bare rock or paved', 'Bare soil or sand', 'Water']
    
    # Network
    filepath_1 = 'weights/CEL_bs8_8cat10channel_ShallowResNeXt/LCZ42_SGD_2_1999'
    filepath_2 = 'weights/8cat10channel_SimpleNet/LCZ42_SGD_1_41999'
    filepath_3 = 'weights/WithDropOut_bs8_sen2_se_resnet50_shallow/LCZ42_SGD_1_15999'

    
    Network_1 = ShallowResNeXt(num_classes = 17, depth = 11, cardinality = 16)
    net_1 = torch.nn.DataParallel(Network_1, device_ids=[0])
    cudnn.benchmark = True    
    Network_1.load_state_dict(torch.load(filepath_1 + '.pth'))
    net_1.eval()
    """
    Network_1 = SimpleNet(17)
    net_1 = torch.nn.DataParallel(Network_1, device_ids=[0])
    cudnn.benchmark = True    
    Network_1.load_state_dict(torch.load(filepath_1 + '.pth'))
    net_1.eval()
    """

    Network_2 = SimpleNet(17)
    net_2 = torch.nn.DataParallel(Network_2, device_ids=[0])
    cudnn.benchmark = True
    Network_2.load_state_dict(torch.load(filepath_2 + '.pth'))
    net_2.eval()

    
    Network_3 = se_resnet50_shallow(17, None)
    net_3 = torch.nn.DataParallel(Network_3, device_ids=[0])
    cudnn.benchmark = True
    Network_3.load_state_dict(torch.load(filepath_3 + '.pth'))
    net_3.eval()

    # initialisation the random forest and load the weight
    clf = RandomForestClassifier(n_estimators = 40, max_features = 'log2')
    clf = joblib.load('40_estimator_max_features_log2_RandomForest.pkl')

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

    PW = torch.nn.functional.normalize(torch.cat((PreciWeight_1.view(17,-1), PreciWeight_2.view(17,-1), PreciWeight_3.view(17,-1)), 1), 1)   #concat two weight and normalise them
    result_tensor = []
    result_permodel = [[],[],[],[]]
    results = { 'Pred':[],  'Conf':[],  'Anno':[]}
    results_underThresh = { 'Pred_RF':[], 'Pred':[],  'Anno':[]}
    results_anno = []
    for i, (Input_sen1, Input_sen2, sen2nonstd, Anno) in enumerate(Dataloader_validation):
        Input_sen1 = Input_sen1.cuda()
        Input_sen2 = Input_sen2.cuda()
        sen2nonstd = sen2nonstd.cuda()

        preds_1 = net_1.forward(Input_sen1, Input_sen2)
        #preds_1normal = torch.nn.functional.normalize(preds_1)
        conf_1, pred_1 = preds_1.data.topk(1, 1, True, True)
        result_permodel[1].append(pred_1.item())

        preds_2 = net_2.forward(Input_sen1, Input_sen2)
        #preds_2normal = torch.nn.functional.normalize(preds_2)
        conf_2, pred_2 = preds_2.data.topk(1, 1, True, True)
        result_permodel[2].append(pred_2.item())

        preds_3 = net_3.forward(sen2nonstd)
        #preds_3normal = torch.nn.functional.normalize(preds_3)
        conf_3, pred_3 = preds_3.data.topk(1, 1, True, True)
        result_permodel[3].append(pred_3.item())

        preds = torch.nn.functional.normalize(preds_1)*PW[:, 0].float().cuda() + torch.nn.functional.normalize(preds_2)*PW[:, 1].float().cuda() + torch.nn.functional.normalize(preds_3)*PW[:, 2].float().cuda()
        RF_Pred = clf.predict(preds[0].detach().cpu().numpy().reshape(1,-1)).tolist()
        #preds = preds_1normal*PW[:, 0].float().cuda() + preds_1normal*PW[:, 1].float().cuda() + preds_3normal*PW[:, 2].float().cuda()
        #result_tensor.append(preds[0].detach().cpu().numpy().tolist())
        conf, pred = preds.data.topk(1, 1, True, True)

        results_anno.append(Anno)
        #append prediction results
        results['Pred'].append(pred.item())#RF_Pred[0])
        results['Conf'].append(conf.item())
        results['Anno'].append(Anno)
        '''
        if conf_select.item() > 0.5:
            results['Pred'].append(pred_select.item())
        else :
            results['Pred'].append(pred.item())
        '''       
        if(i%10000 == 0):
            print(i)
        #append annotation results
        #print(conf.item())
        if conf > 0.7:
            results_underThresh['Pred'].append(pred.item())
            results_underThresh['Pred_RF'].append(RF_Pred[0])            
            #append annotation results
            results_underThresh['Anno'].append(Anno.item())

    #np.save('Realresult_tensor.npy',result_tensor)
    #np.save('Real_training_results_anno.npy', results_anno)
    #np.save('results.npy', results)
    print(accuracy_score(results_underThresh['Pred'], results_underThresh['Anno'], normalize = True))
    print(accuracy_score(results_underThresh['Pred_RF'], results_underThresh['Anno'], normalize = True))
    print(accuracy_score(result_permodel[1], result_permodel[2], normalize = True))
    print(len(results_underThresh['Pred']), len(Dataloader_validation)) 
    
    print('Accuracy under threshold: %0.6f'%(accuracy_score(results_underThresh['Pred'], results_underThresh['Anno'], normalize = True)))

    print('Accuracy: %0.6f'%(accuracy_score(results['Pred'], results['Anno'], normalize = True)))
    cnf_matrix = confusion_matrix(results['Anno'], results['Pred'])
    #np.save('results/confmat_Merge.npy',cnf_matrix)
    #np.set_printoptions(precision=2)
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
        
