from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
import torch
import torch.nn.functional as F
import os
import pickle

from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, top_k_accuracy_score
from .util import _to_1d_binary, insert_SQL 
from .agree import find_optimal_svm
from utils.inception import InceptionModel
import copy
from torch.autograd import Variable

from sktime.datatypes._panel._convert import from_2d_array_to_nested

def train_single(epoch, train_loader, model, optimizer, config):
    model.train()

    for idx, data in enumerate(train_loader):
        input, target = data
        input = input.float()
        input = input.to(config.device)
        target = target.to(config.device)
        feat_s, logit_s = model(input)
              
        if len(target.shape) == 1:
            loss = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def train_distilled(epoch, train_loader, module_list, criterion_list, optimizer, config, teacher_probs, teacher_list):

    for module in module_list:
        module.eval()

    module_list[0].train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_weights = module_list[-1]

    total_kl = 0
    total_ce_loss = 0
        
    total_teacher_losses = np.empty(config.teachers)

    for idx, data in enumerate(train_loader):
        batch_loss = 0
        teachers_loss = torch.zeros(config.teachers, dtype=torch.float32, device = config.device)
        reward = torch.zeros(config.teachers, dtype=torch.float32, device = config.device)
        
        input, target = data
        index = len(input)

        input = input.float()

        input = input.to(config.device)
        target = target.to(config.device)

        feat_s, logit_s = model_s(input)

        if len(target.shape) == 1:
            loss_cls = criterion_cls(logit_s, target.unsqueeze(-1).float())
        else:
            loss_cls = criterion_cls(logit_s, target.argmax(dim=-1))
        
        if config.evaluation == 'lightts' or config.evaluation == 'aed-loo':
            for teacher in range(0,config.teachers):
                
                if config.teacher_type == 'Inception':
                    model_t = module_list[teacher+1]
                    with torch.no_grad():
                        feat_t, logit_t = model_t(input)
                else:
                    logit_t = torch.as_tensor(teacher_list[teacher], dtype = torch.float, device = config.device)

                loss_div = criterion_div(logit_s, logit_t)
                teachers_loss[teacher] += loss_div
                batch_loss += loss_div
                
            teacher_losses, ensemble_weights = model_weights(teachers_loss,train=True)
            ensemble_loss = torch.sum(teacher_losses)
            
        elif config.evaluation == 'classic':
            logit_list = []
            for teacher in range(0,config.teachers):
                if config.teacher_type == 'Inception':
                    model_t = module_list[teacher+1]

                    with torch.no_grad():
                        feat_t, logit_t = model_t(input)
                        logit_list.append(logit_t)
                else:
                    logit_t = torch.as_tensor(teacher_list[teacher], dtype = torch.float, device = config.device)
                    logit_list.append(logit_t)

            loss_div = criterion_div(logit_s, logit_list)
            batch_loss += loss_div
            ensemble_loss = batch_loss
            

        elif config.evaluation == 'agree':
            logit_t_list = []
            
            for teacher in range(0,config.teachers):
                if config.teacher_type == 'Inception':
                    model_t = module_list[teacher+1]
                    with torch.no_grad():
                        feat_t, logit_t = model_t(input)
                        logit_t_list.append(logit_t)
                else:
                    logit_t = torch.as_tensor(teacher_list[teacher], dtype = torch.float, device = config.device)
                    logit_t_list.append(logit_t)
                
            loss_div_list = []
            grads = []
            logit_s.register_hook(lambda grad: grads.append(
                Variable(grad.data.clone(), requires_grad=False)))
            for logit_t in logit_t_list:
                optimizer.zero_grad()
                loss_s = criterion_div(logit_s, logit_t)
                loss_s.backward(retain_graph=True)
                loss_div_list.append(loss_s)

            if len(target.shape) == 1: 
                scale = find_optimal_svm(torch.stack(grads),device=config.device,is_norm=False)
            else:
                scale = find_optimal_svm(torch.stack(grads),device=config.device)
            
            losses_div_tensor = torch.stack(loss_div_list)

            scale = scale.to(config.device)
            losses_div_tensor.to(config.device)
            ensemble_loss = torch.dot(scale, losses_div_tensor)
            
        elif config.evaluation == 'cawpe':
            logit_t_list = []
            
            for teacher in range(0,config.teachers):
                if config.teacher_type == 'Inception':
                    model_t = module_list[teacher+1]
                    with torch.no_grad():
                        feat_t, logit_t = model_t(input)
                        logit_t_list.append(logit_t)
                else:
                    X_test = from_2d_array_to_nested(input.squeeze().cpu().detach().numpy())
                    logit_t_np = teacher_list[teacher].predict_proba(X_test)
                    logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                    logit_t_list.append(logit_t)
                        
            loss_div_list = []
            grads = []
            logit_s.register_hook(lambda grad: grads.append(
                Variable(grad.data.clone(), requires_grad=False)))
            for logit_t in logit_t_list:
                optimizer.zero_grad()
                loss_s = criterion_div(logit_s, logit_t)
                loss_s.backward(retain_graph=True)
                loss_div_list.append(loss_s)

            scale = teacher_probs
            losses_div_tensor = torch.stack(loss_div_list)

            scale = scale.to(config.device)
            losses_div_tensor.to(config.device)
            ensemble_loss = torch.sum(torch.mul(scale, losses_div_tensor)) #torch.dot(scale, losses_div_tensor)
            
        elif config.evaluation == 'reinforced':
            for teacher in range(0,config.teachers):
                
                if teacher_probs[teacher] >= 1/config.teachers:
                    if config.teacher_type == 'Inception':
                        model_t = module_list[teacher+1]
                        with torch.no_grad():
                            feat_t, logit_t = model_t(input)
                    else:
                        logit_t = torch.as_tensor(teacher_list[teacher], dtype = torch.float, device = config.device)
                            
                    if len(target.shape) == 1:
                        loss_ce = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
                    else:
                        loss_ce = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')    

                    loss_div = criterion_div(logit_s, logit_t)
                    teachers_loss[teacher] += loss_div
                    reward[teacher] += loss_div + loss_ce

                    batch_loss += loss_div

            ensemble_loss = torch.sum(teachers_loss)
                        
        if len(target.shape) == 1:
            loss_cls = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss_cls = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')       

        if config.w_kl == -1:
            loss = config.w_ce * loss_cls + config.teachers * ensemble_loss
        else:
            loss = config.w_ce * loss_cls + config.w_kl * ensemble_loss

        total_kl += batch_loss
        total_ce_loss += loss_cls
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return reward


def validation(epoch, val_loader, module_list, criterion_list, optimizer, config, t_list = 0):

    for module in module_list:
        module.eval()

    module_list[-1].train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_weights = module_list[-1]

    total_kl = 0
    total_ce_loss = 0
        
    total_teacher_losses = np.empty(config.teachers)

    for idx, data in enumerate(val_loader):
        batch_loss = 0
        teachers_loss = torch.zeros(config.teachers, dtype=torch.float32, device = config.device)
        
        input, target = data
        index = len(input)

        input = input.float()

        input = input.to(config.device)
        target = target.to(config.device)

        with torch.no_grad():
            feat_s, logit_s = model_s(input)

        if len(target.shape) == 1:
            loss_cls = criterion_cls(logit_s, target.unsqueeze(-1).float())
        else:
            loss_cls = criterion_cls(logit_s, target.argmax(dim=-1))
        
        for teacher in range(0,config.teachers):
            if config.teacher_type == 'Inception':
                model_t = module_list[teacher+1]
                with torch.no_grad():
                    feat_t, logit_t = model_t(input)
            else:
                logit_t = torch.as_tensor(teacher_list[teacher], dtype = torch.float, device = config.device)
                    
            loss_div = criterion_div(logit_s, logit_t)
            teachers_loss[teacher] += loss_div
            batch_loss += loss_div
                    
        if len(target.shape) == 1:
            loss_cls = F.binary_cross_entropy_with_logits(logit_s, target.unsqueeze(-1).float(), reduction='mean')
        else:
            loss_cls = F.cross_entropy(logit_s, target.argmax(dim=-1), reduction='mean')       

        teacher_losses, ensemble_weights = model_weights(teachers_loss,train=False)
        ensemble_loss = torch.sum(teacher_losses)

        if config.w_kl == -1:
            loss = config.w_ce * loss_cls + config.teachers * ensemble_loss
        else:
            loss = config.w_ce * loss_cls + config.w_kl * ensemble_loss

        total_kl += batch_loss
        total_ce_loss += loss_cls
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    config.teacher_weights = ensemble_weights.tolist()
    return ensemble_weights.tolist()


def evaluate(test_loader, model, config, epochs=0, training_time=0):
    model.eval()

    start_test = time.time()
    with torch.no_grad():
        true_list, preds_list = [], []
        for x, y in test_loader:
            x, y = x.to(config.device), y.to(config.device)
            with torch.no_grad():
                true_list.append(y.cpu().detach().numpy())
                _, preds = model(x)
                if len(y.shape) == 1:
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.softmax(preds, dim=-1)
                preds_list.append(preds.cpu().detach().numpy())
        testing_time = time.time() - start_test
        true_np, preds_np = np.concatenate(true_list), np.concatenate(preds_list)
        accuracy = accuracy_score(*_to_1d_binary(true_np, preds_np), normalize=True)
        true_1d,_ = _to_1d_binary(true_np, preds_np)

        if config.num_classes > 5:
            try:
                accuracy_5 = top_k_accuracy_score(true_1d, preds_np, normalize=True, k=5)
                roc_auc = roc_auc_score(true_np, preds_np)
                pr_auc = average_precision_score(true_np, preds_np)
            except Exception as e:
                accuracy_5 = roc_auc = pr_auc = -1
        else:
            accuracy_5 = roc_auc = pr_auc = -1


        if training_time == -1:
            return accuracy
        else:
            type_q = str(config.layer1) + "(" + str(config.bit1) + ")-" + str(config.layer2) + "(" + str(config.bit2) + ")-" + str(config.layer2) + "(" + str(config.bit3) + ")"

            #insert_SQL(config.teacher_type, config.pid, config.dataset, "Gumble", config.gumbel, type_q, config.teachers,
            #           config.evaluation, accuracy, "Top 5", accuracy_5, "Epochs", epochs, "Training time", 
            #           training_time,"Test time",testing_time)
            print("Epoch: " + str(epochs) + ", Accuracy: " + str(accuracy))
            return accuracy

def evaluate_ensemble(test_loader, config):
    
    teachers = [i for i in range(0,config.teachers)]
    ensemble_result = []
    if config.teacher_type == 'Inception':
        for teacher in teachers:
            savepath = Path('./teachers/Inception_' + config.dataset + '_' + str(teacher) + '_teacher.pkl')
            teacher_config = copy.deepcopy(config)
            teacher_config.bits = 32
            teacher_config.bit1 = teacher_config.bit2 = teacher_config.bit3 = teacher_config.bits
            teacher_config.layer1 = teacher_config.layer2 = teacher_config.layer3 = 3
            model_t = InceptionModel(num_blocks=3, in_channels=1, out_channels=[10,20,40],
                           bottleneck_channels=32, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=config.num_classes,config=teacher_config)

            model_t.load_state_dict(torch.load(savepath, map_location=config.device))
            model_t.eval()
            model_t = model_t.to(config.device)

            
            with torch.no_grad():
                true_list, preds_list = [], []
                for x, y in test_loader:
                    x, y = x.to(config.device), y.to(config.device)
                    with torch.no_grad():
                        true_list.append(y.cpu().detach().numpy())
                        _, preds = model_t(x)
                        if len(y.shape) == 1:
                            preds = torch.sigmoid(preds)
                        else:
                            preds = torch.softmax(preds, dim=-1)
                        preds_list.append(preds)

                true_np, preds_tensor = np.concatenate(true_list), torch.cat(preds_list)
            ensemble_result.append(preds_tensor)

    else:
        for teacher in teachers:
            savepath = Path('./teachers/'+ config.teacher_type + '_' + config.dataset + '_' + str(teacher) + '_teacher.pkl')
            with open(savepath,'rb') as file:
                pickle_saved = pickle.load(file)
            true_list, preds_list = [], []
            
            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                true_list.append(y.cpu().detach().numpy())
                X_test = from_2d_array_to_nested(x.squeeze().cpu().detach().numpy())
                logit_t_np = pickle_saved.predict_proba(X_test)
                logit_t = torch.as_tensor(logit_t_np, dtype = torch.float, device = config.device)
                
                if len(y.shape) == 1:
                    preds = torch.sigmoid(logit_t)
                else:
                    preds = torch.softmax(logit_t, dim=-1)
                preds_list.append(preds)
            true_np, preds_tensor = np.concatenate(true_list), torch.cat(preds_list)
            sum_np = preds_tensor.cpu().detach().numpy()
            accuracy = accuracy_score(*_to_1d_binary(true_np, sum_np), normalize=True)
            true_1d,_ = _to_1d_binary(true_np, sum_np)
            try:
                accuracy_5 = top_k_accuracy_score(true_1d, sum_np, normalize=True, k=5)
            except Exception as e: # Undefinition for few classes
                accuracy_5 = -1
                
            type_q = "Full precision"
            #insert_SQL(config.teacher_type, config.pid, config.dataset, "Teacher", teacher, type_q, 32, config.evaluation,
            #           accuracy, "Top 5", accuracy_5, "Metric 2", 0, "Metric 3", 0, "Metric 4", 0) 
            print("Teacher accuracy: " + str(accuracy))
            ensemble_result.append(preds_tensor)
        
    sum_probabilities = torch.stack(ensemble_result).sum(dim=0)
    sum_np = sum_probabilities.cpu().detach().numpy()

    accuracy = accuracy_score(*_to_1d_binary(true_np, sum_np), normalize=True)
    true_1d,_ = _to_1d_binary(true_np, sum_np)
    
    if config.num_classes > 5:
        try:
            accuracy_5 = top_k_accuracy_score(true_1d, sum_np, normalize=True, k=5)
        except Exception as e: # Undefinition for few classes
            accuracy_5 = -1
    else:
        accuracy_5 = -1
        
    type_q = "Full precision"
    #insert_SQL(config.teacher_type, config.pid, config.dataset, "Teacher Ensemble", 0, type_q, 32, config.evaluation,
    #           accuracy, "Top 5", accuracy_5, "Metric 2", 0, "Metric 3", 0, "Metric 4", 0) 
    print("Ensemble accuracy: " + str(accuracy))
    return accuracy
