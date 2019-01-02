"""
Training Network by int type Annotation input. This is designed for loss like CrossEntropyLoss
"""
import torch
import time
import h5py
import math
import scipy.stats as ss
import scipy.misc as sm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import torch.nn.init as init
from loss.DistillationLoss import DistillationLoss
from loss.SmoothingLoss import SmoothingLoss
from loss.focalloss import FocalLoss
from torch.autograd import Variable
from basenet.NetworkFactory import NetworkFactory
from H5Dataset import H5Dataset, H5DatasetSia, H5DatasetSoftAnno, H5DatasetSiaResample

import argparse

parser = argparse.ArgumentParser(description = 'Tiangong')
parser.add_argument('--dataset_root', default = 'data', type = str)
parser.add_argument('--adaptive_by_confmat', default = None, type = str)
parser.add_argument('--class_num', default = 17, type = int)
parser.add_argument('--batch_size', default = 8, type = int)
parser.add_argument('--num_workers', default = 1, type = int)
parser.add_argument('--start_iter', default = 0, type = int)
parser.add_argument('--adjust_iter', default = 40000, type = int)
parser.add_argument('--end_iter', default = 60000, type = int)
parser.add_argument('--lr', default = 0.002, type = float)
parser.add_argument('--momentum', default = 0.9, type = float)
parser.add_argument('--weight_decay', default = 5e-4, type = float)
parser.add_argument('--gamma', default = 0.1, type = float)
parser.add_argument('--resume', default = None, type = str)
parser.add_argument('--warmup_cosineshedule', default = None, type = str)
parser.add_argument('--basenet', default = 'SimpleNet', type = str)
parser.add_argument('--MultiLabel', default = None, type = str)
parser.add_argument('--AddNoise', default = True, type = bool)
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')

args = parser.parse_args()       #global variable

def main():
    #create model
    best_prec1 = 0
    
    idxs = np.load('data/MulResampleIndex.npy')
    Dataset_train = H5DatasetSiaResample(root = args.dataset_root, mode = 'training',indices = idxs,  symmetrize = True)
    Dataloader_train = data.DataLoader(Dataset_train, args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)
    idxs_val_fortrain = np.load('data/MulResampleIndex_forval.npy')
    Dataset_val = H5DatasetSiaResample(root = args.dataset_root, mode = 'validation', indices = idxs_val_fortrain, symmetrize = True)
    Dataloader_validation_fortrain = data.DataLoader(Dataset_val, batch_size = args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)
    
    #idxs_val = np.load('data/rand_tiers_1_val.npy')
    Dataset_val = H5DatasetSia(root = args.dataset_root, mode = 'validation_copy', symmetrize = False)
    Dataloader_validation = data.DataLoader(Dataset_val,  batch_size = 1,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    '''
    idxs_val = np.load('data/MulResampleIndex.npy')
    Dataset_train = H5DatasetSoftAnno(root = args.dataset_root, mode = 'training', indices = idxs_val)
    Dataloader_train = data.DataLoader(Dataset_train, args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)
    
    Dataset_val = H5DatasetSia(root = args.dataset_root, mode = 'validation')
    Dataloader_validation = data.DataLoader(Dataset_val, batch_size = 1,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    idxs_val = np.load('data/MulResampleIndex_forval.npy')
    Dataset_validation = H5DatasetSoftAnno(root = args.dataset_root, mode = 'validation', indices = idxs_val)
    Dataloader_validation_fortrain = data.DataLoader(Dataset_validation, batch_size = args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)
    '''
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)

    model = NetworkFactory.ConsturctNetwork(args.basenet, args.resume).cuda()
    model = model.cuda()
    cudnn.benchmark = True

    #weights = torch.FloatTensor(weights)
    if args.MultiLabel == None :
        #criterion = nn.CrossEntropyLoss().cuda()#criterion = DistillationLoss()
         criterion = FocalLoss(gamma = 2, alpha = 0.25) 
    else :
        criterion = nn.MultiLabelMarginLoss().cuda()

    Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, momentum = args.momentum,
                          weight_decay = args.weight_decay, nesterov = True)
    #torch.save(model.state_dict(), 'weights/bs32_8cat10channel_NonNeste_'+ args.basenet +'/'+ 'LCZ42_SGD' + '.pth')
    prefix = 'weights/FL_WarmUp_Cosine_AllBalanced_bs32_8cat10channel_'

    for epoch in range(args.start_epoch, args.epochs):
        
        if epoch > 1 :
            adjust_learning_rate(Optimizer, epoch, mode = 'Cosine', decay = math.sqrt(0.5))
        
        # train for one epoch
        if(epoch % 2 == 0) :
            train(Dataloader_train, model, criterion, Optimizer, epoch, Dataloader_validation, args.AddNoise)   
        else:
            train(Dataloader_validation_fortrain, model, criterion, Optimizer, epoch, Dataloader_validation, args.AddNoise)
        

def train(Dataloader,model, criterion, optimizer, epoch, Dataloader_validation, AddNoise):
    # Priors
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    Noise = False
    if epoch%2 == 0 :
        freq = 3277         # for tiers freq = 2447
    else :
        freq = 902
    model.train()
    model = model.cuda()
    # train
    end = time.time()
    
    if epoch == 0 :
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*(float(freq)/26220)
            print(param_group['lr'])
    
    for param_group in optimizer.param_groups:
        print('Learning Rate of Current Epoch: %0.6f'%param_group['lr'])
    
    for i, (input1, input2, anos) in enumerate(Dataloader):
        #print(i)

        data_time.update(time.time() - end)
        #print(anos.size())
        target = anos.cuda(async=True)

        with torch.no_grad():
            input1_var = Variable(input1.cuda())
            input2_var = Variable(input2.cuda())
            target_var = Variable(target.cuda())

            if epoch < 6 and np.random.rand() > 0.5:
                input1_var = gaussian(input1_var, 0, 0.01)
                input2_var = gaussian(input2_var, 0, 0.01)
                input1_var, input2_var = multiplicative_denoising(input1_var, input2_var)

            #softanos = Variable(softanos.cuda())
        # compute output
        output = model(input1_var.cuda(), input2_var.cuda())
        #print(output,target_var)
        #loss = criterion(output, softanos, target_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input1.size(0))
        top1.update(prec1[0], input1.size(0))
        top5.update(prec5[0], input1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i+1, len(Dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            batch_time.reset()
            data_time.reset()
            losses.reset()
            top1.reset()
            top5.reset()
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr*(float(i + 1 + freq)/26220)
                    print(param_group['lr'])
  
        if (i+1) % freq == 0:
            #Return_Confmatrix(model, epoch, Dataloader_validation, itera = i)
            if not os.path.exists('weights/FL_WarmUp_Cosine_AllBalanced_bs32_8cat10channel_' + args.basenet ):
                os.mkdir('weights/FL_WarmUp_Cosine_AllBalanced_bs32_8cat10channel_' + args.basenet)
            torch.save(model.state_dict(), 'weights/FL_WarmUp_Cosine_AllBalanced_bs32_8cat10channel_' + args.basenet +'/'+ 'LCZ42_SGD_' + repr(epoch)+ '_' + repr(i) + '.pth')
            model.train()
            '''
            if (i+1)%(freq*4) == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*math.sqrt(0.5)
                    print(param_group['lr'])
            '''

def Return_Confmatrix(net, epoch, Dataloader, itera):
    """
    input: neural network, epoch number
    output: reverse of diagram of confusion matrix to serve as the weight of loss.
    
    fid = h5py.File('data/validation.h5')
    Dataset = fid['sen2']
    """
    net.eval()

    results = []
    results_anno = []

    for i, (Input_sen1, Input_sen2, Anno) in enumerate(Dataloader):
        Input_sen1 = Input_sen1.cuda()
        Input_sen2 = Input_sen2.cuda()
        preds = net.forward(Input_sen1, Input_sen2)
        _, pred = preds.data.topk(1, 1, True, True)
        results.append(pred.item())
        results_anno.append(Anno)
    cnf_matrix = confusion_matrix(results_anno, results)
    np.save('weights/FL_WarmUp_Cosine_AllBalanced_bs32_8cat10channel_' + args.basenet +'/'+ 'LCZ42_SGD_' + repr(epoch) + '_' + repr(itera) + '.npy', cnf_matrix)
    cnf_tr = np.trace(cnf_matrix)
    cnf_tr = cnf_tr.astype('float')
    print("The accuracy in Validation set of " + repr(epoch)  + '_' + repr(itera) + " epoch : " + repr(cnf_tr/len(Dataloader))+"%")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, mode, decay = 0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if mode == 'Exponential' : 
        lr = args.lr * (decay ** (epoch // 1)) # former 0.95 per epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif mode == 'Cosine' :
        lr = args.lr * (0.5 * (1 + math.cos( (epoch - 1) * math.pi / 20 ))) # former 0.95 per epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif mode == 'Linear' :
        lr = args.lr * (float(epoch+1)/5) # former 0.95 per epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_learning_rate_Cosine(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 * (1 + math.cos( (epoch - 1) * math.pi / 20 ))) # former 0.95 per epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_Linear(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (float(epoch+1)/5) # former 0.95 per epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)                                           # max range for candidate prediction. like topk=(1, 5, 7), maxk =5
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))      # target (1,8) , pred(5, 8) see if 
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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

if __name__ == '__main__':
    main()
    
