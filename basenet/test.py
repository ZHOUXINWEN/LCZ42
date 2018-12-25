import torch
import torch.nn as nn
from resnext import CifarResNeXt, ShallowResNeXt
from densenet import DenseNet
from SimpleNet import SimpleNet
import torch.backends.cudnn as cudnn
torch.set_default_tensor_type('torch.cuda.FloatTensor')
cudnn.benchmark = True
#model = SimpleNet(17).cuda()
#model = CifarResNeXt(num_classes = 17, depth = 29, cardinality = 8).cuda()
#model = ShallowResNeXt(num_classes = 17, depth = 11, cardinality = 16, UseGN = True) 
model = DenseNet(num_classes = 17).cuda()   
print(model(torch.randn(32,8,32,32).float().cuda(),torch.randn(32,10,32,32).float().cuda()))
