import os, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.svm import OneClassSVM as ssvm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from utils.data import get_kfold_loaders
import torch, copy, time, numpy as np
from utils.trainer import train_single, evaluate
from utils.inception import InceptionModel

def train_probabilities(config):

    teacher_config = copy.deepcopy(config)
    teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = 32
    teacher_config.layer1 = teacher_config.layer2 = teacher_config.layer3 = 3
    model = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                   bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                   num_pred_classes=config.num_classes,config=teacher_config)
    model = model.to(config.device)
    teacher_prob = np.zeros(config.teachers) 
    
    for teacher in range(0,config.teachers):
        config.init_seed = teacher
        np.random.seed(teacher)
        torch.manual_seed(teacher)
        torch.cuda.manual_seed(teacher)
        torch.backends.cudnn.deterministic = True
    
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        train_cross_loaders, test_cross_loaders = get_kfold_loaders(config)
        cross_accuracy = np.zeros(config.cross_validation) 
        start_training = time.time()

        for cross_val in range(0,config.cross_validation):
            for epoch in range(1, 3): #101
                train_single(epoch, train_cross_loaders[cross_val], model, optimizer, config)
            training_time = -1
            cross_accuracy[cross_val] = evaluate(test_cross_loaders[cross_val], model, config, epoch, training_time)
        teacher_prob[teacher] = np.max(cross_accuracy)
        
    return F.softmax(torch.tensor(teacher_prob,device=config.device),dim=0)