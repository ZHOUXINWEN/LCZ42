from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['densenet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        #self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(8, planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        #self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn1 = nn.GroupNorm(8, inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=10, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=32, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        '''
        -------------------------------------------------------------------------------
        initialize subnetwork for sen1 input
        -------------------------------------------------------------------------------
        '''
        self.inplanes_sen1 = growthRate * 2                 # here self.inplanes is 24
        self.conv1_sen1 = nn.Conv2d(8, self.inplanes_sen1, kernel_size=3, padding=1,
                               bias=False)

        self.dense1_sen1 = self._make_denseblock(block, n, mode = 'sen1')
        self.trans1_sen1 = self._make_transition(compressionRate, mode = 'sen1')
        self.dense2_sen1 = self._make_denseblock(block, n, mode = 'sen1')
        self.trans2_sen1 = self._make_transition(compressionRate, mode = 'sen1')
        self.dense3_sen1 = self._make_denseblock(block, n, mode = 'sen1')

        #self.bn_sen1 = nn.BatchNorm2d(self.inplanes_sen1)
        self.bn_sen1 = nn.GroupNorm(8, self.inplanes_sen1)
        self.relu_sen1 = nn.ReLU(inplace=True)

        '''
        -------------------------------------------------------------------------------
        initialize subnetwork for sen2 input
        -------------------------------------------------------------------------------
        '''

        self.inplanes_sen2 = growthRate * 2                 # here self.inplanes is 24 
        self.conv1_sen2 = nn.Conv2d(10, self.inplanes_sen2, kernel_size=3, padding=1,
                               bias=False)

        self.dense1_sen2 = self._make_denseblock(block, n, mode = 'sen2')
        self.trans1_sen2 = self._make_transition(compressionRate, mode = 'sen2')
        self.dense2_sen2 = self._make_denseblock(block, n, mode = 'sen2')
        self.trans2_sen2 = self._make_transition(compressionRate, mode = 'sen2')
        self.dense3_sen2 = self._make_denseblock(block, n, mode = 'sen2')

        #self.bn_sen2 = nn.BatchNorm2d(self.inplanes_sen2)
        self.bn_sen2 = nn.GroupNorm(8, self.inplanes_sen2)
        self.relu_sen2 = nn.ReLU(inplace=True)


        self.avgpool = nn.AvgPool2d(8)
        #print(self.inplanes_sen1, self.inplanes_sen2)
        self.fc = nn.Linear(self.inplanes_sen1 + self.inplanes_sen2, num_classes)
        '''
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''
    def _make_denseblock(self, block, blocks, mode):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            if mode == 'sen1' :
                layers.append(block(self.inplanes_sen1, growthRate=self.growthRate, dropRate=self.dropRate))
                self.inplanes_sen1 += self.growthRate
            else :
                layers.append(block(self.inplanes_sen2, growthRate=self.growthRate, dropRate=self.dropRate))
                self.inplanes_sen2 += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, mode):
        if mode == 'sen1' :
            inplanes = self.inplanes_sen1
            outplanes = int(math.floor(self.inplanes_sen1 // compressionRate))
            self.inplanes_sen1 = outplanes
        else :
            inplanes = self.inplanes_sen2
            outplanes = int(math.floor(self.inplanes_sen2 // compressionRate))
            self.inplanes_sen2 = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x_sen1, x_sen2):
        x_sen1 = self.conv1_sen1(x_sen1)
        x_sen1 = self.trans1_sen1(self.dense1_sen1(x_sen1)) 
        x_sen1 = self.trans2_sen1(self.dense2_sen1(x_sen1)) 
        x_sen1 = self.dense3_sen1(x_sen1)
        x_sen1 = self.bn_sen1(x_sen1)
        x_sen1 = self.relu_sen1(x_sen1)

        x_sen2 = self.conv1_sen2(x_sen2)
        x_sen2 = self.trans1_sen2(self.dense1_sen2(x_sen2)) 
        x_sen2 = self.trans2_sen2(self.dense2_sen2(x_sen2)) 
        x_sen2 = self.dense3_sen2(x_sen2)
        x_sen2 = self.bn_sen2(x_sen2)
        x_sen2 = self.relu_sen2(x_sen2)

        x = torch.cat((x_sen1, x_sen2), 1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class DenseNetSia(nn.Module):

    def __init__(self, depth=10, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=64, compressionRate=2):
        super(DenseNetSia, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        '''
        -------------------------------------------------------------------------------
        initialize subnetwork for sen1 input
        -------------------------------------------------------------------------------
        '''
        self.inplanes_sen1 = growthRate * 2                 # here self.inplanes is 24
        self.conv1_sen1 = nn.Conv2d(8, 64, kernel_size=3, padding=1,
                               bias=False)
        self.conv1_sen2 = nn.Conv2d(10, 64, kernel_size=3, padding=1,
                               bias=False)
        self.dense1_sen1 = self._make_denseblock(block, n, mode = 'sen1')

        self.trans1_sen1 = self._make_transition(compressionRate, mode = 'sen1')

        self.dense2_sen1 = self._make_denseblock(block, n, mode = 'sen1')

        self.trans2_sen1 = self._make_transition(compressionRate, mode = 'sen1')

        self.dense3_sen1 = self._make_denseblock(block, n, mode = 'sen1')


        #self.bn_sen1 = nn.BatchNorm2d(self.inplanes_sen1)
        self.bn_sen1 = nn.GroupNorm(8, self.inplanes_sen1)
        self.relu_sen1 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(8)
        #print(self.inplanes_sen1, self.inplanes_sen2)
        self.fc = nn.Linear(self.inplanes_sen1, num_classes)
        '''
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''
    def _make_denseblock(self, block, blocks, mode):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            if mode == 'sen1' :
                layers.append(block(self.inplanes_sen1, growthRate=self.growthRate, dropRate=self.dropRate))
                self.inplanes_sen1 += self.growthRate
            else :
                layers.append(block(self.inplanes_sen2, growthRate=self.growthRate, dropRate=self.dropRate))
                self.inplanes_sen2 += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, mode):
        if mode == 'sen1' :
            inplanes = self.inplanes_sen1
            outplanes = int(math.floor(self.inplanes_sen1 // compressionRate))
            self.inplanes_sen1 = outplanes
        else :
            inplanes = self.inplanes_sen2
            outplanes = int(math.floor(self.inplanes_sen2 // compressionRate))
            self.inplanes_sen2 = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x_sen1, x_sen2):
        x_sen1 = self.conv1_sen1(x_sen1)
        x_sen2 = self.conv1_sen2(x_sen2)
        x = torch.cat((x_sen1, x_sen2), 1)

        x = self.trans1_sen1(self.dense1_sen1(x)) 
        x = self.trans2_sen1(self.dense2_sen1(x)) 
        x = self.dense3_sen1(x)
        x = self.bn_sen1(x)
        x = self.relu_sen1(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
