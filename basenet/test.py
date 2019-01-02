import torch
import torch.nn as nn

from NetworkFactory import NetworkFactory
import torch.backends.cudnn as cudnn
torch.set_default_tensor_type('torch.cuda.FloatTensor')
cudnn.benchmark = True

model = NetworkFactory.ConsturctNetwork('se_resnext50_32x4d', None).cuda()
print(model(torch.randn(32,8,32,32).float().cuda(),torch.randn(32,10,32,32).float().cuda()))
