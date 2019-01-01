import torch
import torch.nn as nn
import torch.nn.functional as func

class DistillationLoss(nn.Module):
    def __init__(self, Temperature = 5):
        super(DistillationLoss,self).__init__()
        self.T = Temperature

    def forward(self, input_z, target_r, target_p):
        soft_r = func.softmax(target_r/self.T, dim = 1)
        soft_z = func.softmax(func.normalize(input_z)/self.T, dim =1)

        CE_loss = func.cross_entropy(input_z, target_p)#, weight = None, ignore_index = -100, reduction = 'mean')
        Distillation_loss = self.T*self.T*func.kl_div(soft_z,soft_r)
        return CE_loss - Distillation_loss


