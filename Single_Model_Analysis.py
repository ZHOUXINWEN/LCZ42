'''
This python file is to test a single network on dataset and visiualise confusion metrix
'''
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
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from H5Dataset import H5DatasetSia
from basenet.SimpleNet import SimpleNetGN,SimpleNet
from basenet.resnext import CifarResNeXt, ShallowResNeXt
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
    Dataset_validation = H5DatasetSia(root = 'data', mode = 'validation', symmetrize = False)
    Dataloader_validation = data.DataLoader(Dataset_validation, batch_size = 1,
                                 num_workers = 1,
                                 shuffle = True, pin_memory = True)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    class_names = ['Compact high-rise', 'Compact midrise', 'Compact lowrise','Open high-rise', 'Open midrise', 'Open lowrise', 'Lightweight low-rise', 'Large low-rise', 'Sparsely built',
               'Heavy industry', 'Dense trees', 'Scattered trees', 'Bush and scrub', 'Low plants', 'Bare rock or paved', 'Bare soil or sand', 'Water']
    
    # Network
    
    #Network = pnasnet5large(6, None)
    #Network = ResNeXt101_64x4d(17)
    #Network = se_resnet50(17, None)
    Network = ShallowResNeXt(num_classes = 17, depth = 11, cardinality = 16)
    net = torch.nn.DataParallel(Network, device_ids=[0])
    cudnn.benchmark = True
    
    Network.load_state_dict(torch.load('weights/CEL_WarmUp_TrainALlValAllBallance_bs32_8cat10channel_ShallowResNeXt/LCZ42_SGD_5_1803.pth'))  #0.9426593142335917
    net.eval()

    results = []
    results_anno = []
    for i, (Input_sen1, Input_sen2, Anno) in enumerate(Dataloader_validation):
        Input_sen1 = Input_sen1.cuda()
        Input_sen2 = Input_sen2.cuda()

        preds = net.forward(Input_sen1, Input_sen2)
        _, pred = preds.data.topk(1, 1, True, True)
        #append prediction results
        results.append(pred.item())
        #append annotation results
        results_anno.append(Anno)
        if(i %10000 == 0 ):
            print(i)
    #print(f1_score(results_anno, results, class_names, average = 'weighted'))
    cnf_matrix = confusion_matrix(results_anno, results)
    #print(classification_report(results_anno, results, class_names))
    #np.save('weights/5e-4_Weight_sen2_MoreDecay_se_resnet50_shallow/confmat_13.npy',cnf_matrix)
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
        
