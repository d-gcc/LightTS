from pathlib import Path
from dataclasses import dataclass
import numpy as np
import os
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import cast, Any, Dict, List, Tuple, Optional

import torch.quantization.quantize_fx as quantize_fx
import copy
import time

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, \
    OneD_SymbolicAggregateApproximation
from sklearn.metrics import mean_squared_error
from math import sqrt

@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float):
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=None)
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))



def load_ucr_data(config, use_encoder=True) -> Tuple[InputData, InputData]:

    train = np.loadtxt(config.data_folder / config.dataset /f'{config.dataset}_TRAIN.tsv', delimiter='\t')
    test = np.loadtxt(config.data_folder / config.dataset /f'{config.dataset}_TEST.tsv', delimiter='\t')

    if use_encoder:
        encoder = OneHotEncoder(categories='auto', sparse=False)
        y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
        y_test = encoder.transform(np.expand_dims(test[:, 0], axis=-1))
    else:
        y_train = np.expand_dims(train[:, 0], axis=-1)
        y_test = np.expand_dims(test[:, 0], axis=-1)

    if y_train.shape[1] == 2:
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]
    
    original_data = train[:, 1:]
    test_data = test[:, 1:]
    
    train_input = InputData(x=torch.from_numpy(train[:, 1:]).unsqueeze(1).float(),y=torch.from_numpy(y_train))
    test_input = InputData(x=torch.from_numpy(test[:, 1:]).unsqueeze(1).float(),y=torch.from_numpy(y_test))

    return train_input, test_input

def get_loaders(config):

    train_data, test_data = load_ucr_data(config)
    train_data, val_data = train_data.split(config.val_size)

    train_loader = DataLoader(TensorDataset(train_data.x, train_data.y),batch_size=config.batch_size,shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data.x, val_data.y),batch_size=config.batch_size,shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data.x, test_data.y),batch_size=config.batch_size,shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_raw_data(config):

    train_data, test_data = load_ucr_data(config, use_encoder=False)
    
    return train_data, test_data

def get_kfold_loaders(config):
    train_loaders = []
    test_loaders = []
    kfold = KFold(n_splits=config.cross_validation)
    train_data, _ = load_ucr_data(config)
    for fold, (train_index, test_index) in enumerate(kfold.split(train_data.x, train_data.y)):

        x_train_fold = train_data.x[train_index]
        x_test_fold = train_data.x[test_index]
        y_train_fold = train_data.y[train_index]
        y_test_fold = train_data.y[test_index]

        train_loaders.append(DataLoader(TensorDataset(x_train_fold, y_train_fold),batch_size=config.batch_size,shuffle=False))
        test_loaders.append(DataLoader(TensorDataset(x_test_fold, y_test_fold),batch_size=config.batch_size,shuffle=False))
        
    return train_loaders, test_loaders