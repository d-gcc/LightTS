from __future__ import print_function

import torch, copy
import torch.nn as nn
import torch.nn.functional as F

class TeacherWeights(nn.Module):
    def __init__(self, config, weights):
        super(TeacherWeights,self).__init__()
        self.W = torch.nn.Parameter(weights)
        self.config = config
        
    def forward(self, x, train=True):
        if self.config.evaluation == 'aed-loo':
            teacher_loss = torch.multiply(self.W, x)
            return teacher_loss, self.W
        else:
            if train:
                w_nograd = copy.deepcopy(self.W)
                w_nograd.requires_grad = False
#                 if self.config.gumbel > 0:
#                     teacher_gs = F.gumbel_softmax(w_nograd, tau = self.config.gumbel, dim=0)
#                     teacher_weights = F.softmax(teacher_gs.mul(-1), dim=0)
#                 else:
                teacher_weights = F.softmax(w_nograd, dim=0)
                teacher_loss = torch.multiply(teacher_weights, x)
                return teacher_loss, w_nograd
            else:
                x_nograd = copy.deepcopy(x)
                x_nograd.requires_grad = False
                if self.config.gumbel > 0:
                    teacher_gs = F.gumbel_softmax(self.W, tau = self.config.gumbel, dim=0)
                    teacher_weights = F.softmax(teacher_gs.mul(-1), dim=0)
                else:
                    teacher_weights = F.softmax(self.W, dim=0)
                teacher_loss = torch.multiply(teacher_weights, x_nograd)
                return teacher_loss, teacher_weights

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss
    
class DistillKLOne(nn.Module):
    def __init__(self, T):
        super(DistillKLOne, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = y_s/self.T
        p_t = y_t/self.T
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class KDEnsemble(nn.Module):
    def __init__(self, T, device):
        super(KDEnsemble, self).__init__()
        self.T = T
        self.device = device

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        if y_t:
            p_t = torch.zeros(y_t[0].size()).to(self.device)
            for y_ti in y_t:
                p_tf = F.softmax(y_ti/self.T, dim=1)
                p_t = torch.add(p_t,p_tf)
                
        loss = F.kl_div(p_s, p_t/len(y_t), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class KDEnsembleOne(nn.Module):
    def __init__(self, T, device):
        super(KDEnsembleOne, self).__init__()
        self.T = T
        self.device = device

    def forward(self, y_s, y_t):
        p_s = y_s/self.T
        if y_t:
            p_t = torch.zeros(y_t[0].size()).to(self.device)
            for y_ti in y_t:
                p_tf = y_ti/self.T
                p_t = torch.add(p_t,p_tf)
                
        loss = F.kl_div(p_s, p_t/len(y_t), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss