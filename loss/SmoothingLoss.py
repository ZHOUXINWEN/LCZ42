import torch
import torch.nn as nn
import torch.nn.functional as func

class SmoothingLoss(nn.Module):
    def __init__(self, class_num = 17, epsilon = 0.1):
        super(SmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.class_num = class_num

    def forward(self, input_z, target_p):
        soft_z = func.softmax(input_z, dim =1)
        soft_p = target_p*(1.0 - ( (self.class_num*self.epsilon)/(self.class_num - 1))) + self.epsilon/(self.class_num - 1)

        return func.kl_div(input_z, target_p)

