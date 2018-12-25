import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict


class SimpleNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes
        
        layer0_modules_sen1 = [
            ('conv1', nn.Conv2d(8, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen2 = [
            ('conv1', nn.Conv2d(10, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen1.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        layer0_modules_sen2.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))

        layer1_modules = [
            ('conv1', nn.Conv2d(128, 256, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer2_modules = [
            ('conv1', nn.Conv2d(256, 512, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer3_modules = [
            ('conv1', nn.Conv2d(512, 1024, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(1024)),
            ('relu1', nn.ReLU(inplace=True)),
        ]

        self.layer0_sen1 = nn.Sequential(OrderedDict(layer0_modules_sen1))
        self.layer0_sen2 = nn.Sequential(OrderedDict(layer0_modules_sen2))
        self.layer1 = nn.Sequential(OrderedDict(layer1_modules))
        self.layer2 = nn.Sequential(OrderedDict(layer2_modules))
        self.layer3 = nn.Sequential(OrderedDict(layer3_modules))

        self.linear1 = nn.Linear(1024, 256)
        #init.kaiming_normal_(self.linear1.weight)

        self.linear2 = nn.Linear(256, num_classes)
        #init.kaiming_normal_(self.linear2.weight)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        '''
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        '''
    def features(self, x_sen1, x_sen2):
        x_sen1 = self.layer0_sen1(x_sen1)
        x_sen2 = self.layer0_sen2(x_sen2)
        #x = x_sen1 + x_sen2
        #print(x_sen1.size(), x_sen2.size())
        x = torch.cat((x_sen1, x_sen2), 1)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
    def logits(self, input):
        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def features_256(self, input):
        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        return x

    def forward(self, x_sen1, x_sen2):
        x = self.features(x_sen1, x_sen2)
        x = self.logits(x)
        return x

class SimpleNetGN(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleNetGN, self).__init__()
        self.num_classes = num_classes
        
        layer0_modules_sen1 = [
            ('conv1', nn.Conv2d(8, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.GroupNorm(32, 64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.GroupNorm(32, 64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.GroupNorm(32, 64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen2 = [
            ('conv1', nn.Conv2d(10, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.GroupNorm(32, 64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.GroupNorm(32, 64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.GroupNorm(32, 64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen1.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        layer0_modules_sen2.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))

        layer1_modules = [
            ('conv1', nn.Conv2d(128, 256, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.GroupNorm(32, 256)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer2_modules = [
            ('conv1', nn.Conv2d(256, 512, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.GroupNorm(32, 512)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer3_modules = [
            ('conv1', nn.Conv2d(512, 1024, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.GroupNorm(32, 1024)),
            ('relu1', nn.ReLU(inplace=True)),
        ]

        self.layer0_sen1 = nn.Sequential(OrderedDict(layer0_modules_sen1))
        self.layer0_sen2 = nn.Sequential(OrderedDict(layer0_modules_sen2))
        self.layer1 = nn.Sequential(OrderedDict(layer1_modules))
        self.layer2 = nn.Sequential(OrderedDict(layer2_modules))
        self.layer3 = nn.Sequential(OrderedDict(layer3_modules))

        self.linear1 = nn.Linear(1024, 256)
        #init.kaiming_normal_(self.linear1.weight)

        self.linear2 = nn.Linear(256, num_classes)
        #init.kaiming_normal_(self.linear2.weight)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        '''
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        '''
    def features(self, x_sen1, x_sen2):
        x_sen1 = self.layer0_sen1(x_sen1)
        x_sen2 = self.layer0_sen2(x_sen2)
        #x = x_sen1 + x_sen2
        #print(x_sen1.size(), x_sen2.size())
        x = torch.cat((x_sen1, x_sen2), 1)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
    def logits(self, input):
        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def features_256(self, input):
        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        return x

    def forward(self, x_sen1, x_sen2):
        x = self.features(x_sen1, x_sen2)
        x = self.logits(x)
        return x
class SimpleNetLRN(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleNetLRN, self).__init__()
        self.num_classes = num_classes

        layer0_modules_sen1 = [
            ('conv1', nn.Conv2d(8, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen2 = [
            ('conv1', nn.Conv2d(10, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen1.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        layer0_modules_sen2.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))

        layer1_modules = [
            ('conv1', nn.Conv2d(128, 256, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer2_modules = [
            ('conv1', nn.Conv2d(256, 512, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer3_modules = [
            ('conv1', nn.Conv2d(512, 1024, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(1024)),
            ('relu1', nn.ReLU(inplace=True)),
        ]

        self.layer0_sen1 = nn.Sequential(OrderedDict(layer0_modules_sen1))
        self.layer0_sen2 = nn.Sequential(OrderedDict(layer0_modules_sen2))

        self.lrn1 = nn.LocalResponseNorm(2)
        self.lrn2 = nn.LocalResponseNorm(2)

        self.layer1 = nn.Sequential(OrderedDict(layer1_modules))
        self.layer2 = nn.Sequential(OrderedDict(layer2_modules))
        self.layer3 = nn.Sequential(OrderedDict(layer3_modules))
        
        self.linear1 = nn.Linear(1024, 256)
        #init.kaiming_normal_(self.linear1.weight)

        self.linear2 = nn.Linear(256, num_classes)
        #init.kaiming_normal_(self.linear2.weight)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        '''
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        '''
    def features(self, x_sen1, x_sen2):
        x_sen1 = self.layer0_sen1(x_sen1)
        x_sen2 = self.layer0_sen2(x_sen2)
        #x = x_sen1 + x_sen2
        #print(x_sen1.size(), x_sen2.size())
        x_sen1 = self.lrn1(x_sen1)
        x_sen2 = self.lrn2(x_sen2)

        x = torch.cat((x_sen1, x_sen2), 1)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
    def logits(self, input):

        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def forward(self, x_sen1, x_sen2):
        x = self.features(x_sen1, x_sen2)
        x = self.logits(x)
        return x

class SimpleNetSen2(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleNetSen2, self).__init__()
        self.num_classes = num_classes

        layer0_modules = [
            ('conv1', nn.Conv2d(10, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))

        layer1_modules = [
            ('conv1', nn.Conv2d(64, 128, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer2_modules = [
            ('conv1', nn.Conv2d(128, 256, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer3_modules = [
            ('conv1', nn.Conv2d(256, 512, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
        ]

        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = nn.Sequential(OrderedDict(layer1_modules))
        self.layer2 = nn.Sequential(OrderedDict(layer2_modules))
        self.layer3 = nn.Sequential(OrderedDict(layer3_modules))

        self.linear1 = nn.Linear(512, 128)
        #init.kaiming_normal_(self.linear1.weight)

        self.linear2 = nn.Linear(128, num_classes)
        #init.kaiming_normal_(self.linear2.weight)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        '''
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        '''
    def features(self, x_sen2):
        x = self.layer0(x_sen2)
        #x = x_sen1 + x_sen2
        #print(x_sen1.size(), x_sen2.size())
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
    def logits(self, input):

        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def forward(self, x_sen2):
        x = self.features(x_sen2)
        x = self.logits(x)
        return x


class SimplEstNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes

        layer0_modules_sen1 = [
            ('conv1', nn.Conv2d(8, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen2 = [
            ('conv1', nn.Conv2d(10, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
        ]
        """
        layer0_modules_sen1.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        layer0_modules_sen2.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        """
        layer1_modules = [
            ('conv1', nn.Conv2d(128, 256, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer2_modules = [
            ('conv1', nn.Conv2d(256, 512, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer3_modules = [
            ('conv1', nn.Conv2d(512, 1024, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(1024)),
            ('relu1', nn.ReLU(inplace=True)),
        ]

        self.layer0_sen1 = nn.Sequential(OrderedDict(layer0_modules_sen1))
        self.layer0_sen2 = nn.Sequential(OrderedDict(layer0_modules_sen2))
        self.layer1 = nn.Sequential(OrderedDict(layer1_modules))
        self.layer2 = nn.Sequential(OrderedDict(layer2_modules))
        self.layer3 = nn.Sequential(OrderedDict(layer3_modules))

        self.linear1 = nn.Linear(1024, 256)
        init.kaiming_normal_(self.linear1.weight)

        self.linear2 = nn.Linear(256, num_classes)
        init.kaiming_normal_(self.linear2.weight)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def features(self, x_sen1, x_sen2):
        x_sen1 = self.layer0_sen1(x_sen1)
        x_sen2 = self.layer0_sen2(x_sen2)
        #x = x_sen1 + x_sen2
        #print(x_sen1.size(), x_sen2.size())
        x = torch.cat((x_sen1, x_sen2), 1)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def logits(self, input):

        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def forward(self, x_sen1, x_sen2):
        x = self.features(x_sen1, x_sen2)
        x = self.logits(x)
        return x
class SimpleNet4x4(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleNet4x4, self).__init__()
        self.num_classes = num_classes

        layer0_modules_sen1 = [
            ('conv1', nn.Conv2d(8, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen2 = [
            ('conv1', nn.Conv2d(10, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]

        layer0_modules_sen1.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        layer0_modules_sen2.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))

        layer1_modules = [
            ('conv1', nn.Conv2d(128, 256, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        layer2_modules = [
            ('conv1', nn.Conv2d(256, 512, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        '''
        layer3_modules = [
            ('conv1', nn.Conv2d(512, 1024, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(1024)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        '''
        self.layer0_sen1 = nn.Sequential(OrderedDict(layer0_modules_sen1))
        self.layer0_sen2 = nn.Sequential(OrderedDict(layer0_modules_sen2))
        self.layer1 = nn.Sequential(OrderedDict(layer1_modules))
        self.layer2 = nn.Sequential(OrderedDict(layer2_modules))
        #self.layer3 = nn.Sequential(OrderedDict(layer3_modules))

        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, num_classes)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def features(self, x_sen1, x_sen2):
        x_sen1 = self.layer0_sen1(x_sen1)
        x_sen2 = self.layer0_sen2(x_sen2)
        x = torch.cat((x_sen1, x_sen2), 1)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        return x

    def logits(self, input):

        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def forward(self, x_sen1, x_sen2):
        x = self.features(x_sen1, x_sen2)
        x = self.logits(x)
        return x

class SimpleNetLeaky(nn.Module):

    def __init__(self, num_classes=1000):
        super(SimpleNetLeaky, self).__init__()
        self.num_classes = num_classes

        layer0_modules_sen1 = [
            ('conv1', nn.Conv2d(8, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
        ]

        layer0_modules_sen2 = [
            ('conv1', nn.Conv2d(10, 64, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
        ]

        layer0_modules_sen1.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        layer0_modules_sen2.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))

        layer1_modules = [
            ('conv1', nn.Conv2d(128, 256, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(256)),
            ('relu1', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
        ]
        layer2_modules = [
            ('conv1', nn.Conv2d(256, 512, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(512)),
            ('relu1', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
        ]
        layer3_modules = [
            ('conv1', nn.Conv2d(512, 1024, 3, stride=2, padding=1,
                                bias=False)),
            ('bn1', nn.BatchNorm2d(1024)),
            ('relu1', nn.LeakyReLU(negative_slope = 0.1, inplace=True)),
        ]

        self.layer0_sen1 = nn.Sequential(OrderedDict(layer0_modules_sen1))
        self.layer0_sen2 = nn.Sequential(OrderedDict(layer0_modules_sen2))
        self.layer1 = nn.Sequential(OrderedDict(layer1_modules))
        self.layer2 = nn.Sequential(OrderedDict(layer2_modules))
        self.layer3 = nn.Sequential(OrderedDict(layer3_modules))

        self.linear1 = nn.Linear(1024, 256)
        init.kaiming_normal_(self.linear1.weight)

        self.linear2 = nn.Linear(256, num_classes)
        init.kaiming_normal_(self.linear2.weight)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def features(self, x_sen1, x_sen2):
        x_sen1 = self.layer0_sen1(x_sen1)
        x_sen2 = self.layer0_sen2(x_sen2)
        #x = x_sen1 + x_sen2
        #print(x_sen1.size(), x_sen2.size())
        x = torch.cat((x_sen1, x_sen2), 1)
        #print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def logits(self, input):

        x = self.dropout1(input)
        x = input.view(input.size(0), -1)

        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return x

    def forward(self, x_sen1, x_sen2):
        x = self.features(x_sen1, x_sen2)
        x = self.logits(x)
        return x
