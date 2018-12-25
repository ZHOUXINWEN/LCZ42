import torch
import time
import h5py
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from basenet.resnext import Sen2ResNeXt
from basenet.pnasnet import pnasnet5large
from sklearn.metrics import confusion_matrix
from basenet.nasnet_mobile import nasnetamobile
from basenet.senet_shallow import se_resnet50_shallow
from H5Dataset import H5Dataset, H5DatasetCat, H5DatasetResample
from basenet.senet import se_resnext101_32x4d,se_resnet101,se_resnet50
from FolderDataset import FolderDatasetSen1,FolderDatasetSen2,FolderDatasetCat

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
parser.add_argument('--lr', default = 0.008, type = float)
parser.add_argument('--momentum', default = 0.9, type = float)
parser.add_argument('--weight_decay', default = 5e-4, type = float)
parser.add_argument('--gamma', default = 0.1, type = float)
parser.add_argument('--resume', default = None, type = str)
parser.add_argument('--basenet', default = 'se_resnet50_shallow', type = str)
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
#parser.add_argument('--fixblocks', default = 2, type = int)
args = parser.parse_args()

def main():
    #create model
    best_prec1 = 0
    # Dataset
    #idxs = np.load('SubResampleIndex.npy')
    Dataset_train = H5Dataset(root = 'data', mode = 'training', DataType = 'sen2')

    Dataloader_train = data.DataLoader(Dataset_train, args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)
    '''
    Dataloader_train = data.DataLoader(Dataset_train, args.batch_size,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)
    '''
    Dataset_validation = H5Dataset(root = args.dataset_root, mode = 'validation', DataType = 'sen2')
    Dataloader_validation = data.DataLoader(Dataset_validation, batch_size = 1,
                                 num_workers = args.num_workers,
                                 shuffle = True, pin_memory = True)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
    if args.basenet == 'ResNeXt':
        model = Sen2ResNeXt(num_classes = args.class_num, depth = 11, cardinality = 16)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True

    if args.basenet == 'SimpleNetSen2':
        model = SimpleNetSen2(args.class_num)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True

    elif args.basenet == 'se_resnet50':
        model = se_resnet50(args.class_num, None)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True

    elif args.basenet == 'se_resnet50_shallow':
        model = se_resnet50_shallow(args.class_num, None)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True

    elif args.basenet == 'nasnetamobile':
        model = nasnetamobile(args.class_num, None)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True

    elif args.basenet == 'pnasnet':
        model = pnasnet5large(args.class_num, None)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True
        if args.resume:
            model.load_state_dict(torch.load(args.resume))
        else:
            state_dict = torch.load('pnasnet5large-bf079911.pth')
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            model.load_state_dict(state_dict, strict = False)
            init.xavier_uniform_(model.last_linear.weight.data)
            model.last_linear.bias.data.zero_()

    elif args.basenet == 'se_resnet101':
        model = se_resnet101(args.class_num, None)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True
        if args.resume:
            model.load_state_dict(torch.load(args.resume))
        else:
            state_dict = torch.load('se_resnet101-7e38fcc6.pth')
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            model.load_state_dict(state_dict, strict = False)
            init.xavier_uniform_(model.last_linear.weight.data)
            model.last_linear.bias.data.zero_()


    elif args.basenet == 'se_resnext101_32x4d':
        model = se_resnext101_32x4d(args.class_num, None)    
        #net = Networktorch.nn.DataParallel(Network, device_ids=[0])
        cudnn.benchmark = True
        if args.resume:
            model.load_state_dict(torch.load(args.resume))
        else:
            state_dict = torch.load('se_resnext101_32x4d-3b2fe3d8.pth')
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            model.load_state_dict(state_dict, strict = False)
            init.xavier_uniform_(model.last_linear.weight.data)
            model.last_linear.bias.data.zero_()

    model = model.cuda()
    cudnn.benchmark = True

    '''
    
    weights = [ 9.73934491,  2.02034301,  1.55741015,  5.70558317,  2.99272419,
        1.39866818, 15.09911288,  1.25512384,  3.63361307,  4.12907813,
        1.1505058 ,  5.18803868,  5.38559738,  1.1929091 , 20.63503344,
        6.24955685,  1.        ]
    '''
    #weights = np.load('Precision_CM.npy')
    #weights = 1/torch.diag(torch.FloatTensor(weights))weight = weights
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = nn.CosineEmbeddingLoss().cuda()
    Optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, momentum = args.momentum,
                          weight_decay = args.weight_decay, nesterov = True)
    torch.save(model.state_dict(), 'weights/lr8e-3_bs8_sen2_'+ args.basenet +'/'+ 'LCZ42_SGD' + '.pth')
    for epoch in range(args.start_epoch, args.epochs):

        #adjust_learning_rate(Optimizer, epoch)
        # train for one epoch
        train(Dataloader_train, model, criterion, Optimizer, epoch, Dataloader_validation)    #train(Dataloader_train, Network, criterion, Optimizer, epoch)


def train(Dataloader,model, criterion, optimizer, epoch, Dataloader_validation):
    # Priors
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    model.train()
    model = model.cuda()

    end = time.time()
    for i, (input,anos) in enumerate(Dataloader):

        data_time.update(time.time() - end)

        target = anos.cuda(async=True)

        with torch.no_grad():
            input_var = Variable(input.cuda())
            target_var = Variable(target.cuda())

        # compute output
        output = model(input_var.cuda(), )
        #print(output,target_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 2000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i+1, len(Dataloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
       # save and test model for half epoch 
        if (i+1) % 2000 == 0:
            weights = Return_Confmatrix(model, epoch, Dataloader_validation, itera = i)
            torch.save(model.state_dict(), 'weights/lr8e-3_bs8_sen2_'+ args.basenet +'/'+ 'LCZ42_SGD_' + repr(epoch)+ '_' + repr(i) + '.pth')
            model.train()
            """
            if ((i+1)%22000 == 0) and (epoch < 2) :
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5
            """
def validate(val_loader,model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            input_var = Variable(input.cuda())
            target_var = Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % 50 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i+1, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def Return_Confmatrix(net, epoch, Dataloader, itera):
    """
    input: neural network, epoch number
    output: reverse of diagram of confusion matrix to serve as the weight of loss.
    """
    fid = h5py.File('data/validation.h5')
    Dataset = fid['sen2']

    net.eval()

    results = []
    results_anno = []

    for i, (Input,Anno) in enumerate(Dataloader):
        Input = Input.cuda()
        preds = net.forward(Input)
        _, pred = preds.data.topk(1, 1, True, True)
        #append prediction results
        results.append(pred.item())
        #append annotation results
        results_anno.append(Anno)
    cnf_matrix = confusion_matrix(results_anno, results)
    np.save('weights/lr8e-3_bs8_sen2_'+ args.basenet +'/'+ 'LCZ42_SGD_' + repr(epoch) + '_' + repr(itera) + '.npy', cnf_matrix)
    cnf_tr = np.trace(cnf_matrix)
    cnf_tr = cnf_tr.astype('float')
    print("The accuracy in Validation set of " + repr(epoch)  + '_' + repr(itera) + " epoch : " + repr(cnf_tr/len(Dataset))+"%")
    '''
    fid.close()
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]           # normalise confusion matrix
    cnfmat_tensor = torch.from_numpy(cnf_matrix).float()                                      # transform from numpy array to pytorch tensor
    weights = 1/torch.diag(cnfmat_tensor)                                                     # abstract recall from conf mat and reverse them.
    return weights
    '''
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.8 ** (epoch // 1)) # former 0.95 per epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
    
