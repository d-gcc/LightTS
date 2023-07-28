#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse, torch, random, pathlib
import numpy as np, pandas as pd
from teacher import RunTeacher
from student import RunStudent
from utils.util import str2bool, get_free_device

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="A dummy argument for Jupyter", default="1")
    parser.add_argument('--dataset', type=str, default='BeetleFly')
    parser.add_argument('--bit1', type=int, default=13)
    parser.add_argument('--bit2', type=int, default=12)
    parser.add_argument('--bit3', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_w', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--init_seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--experiment', type=str, default='student', choices=['teacher', 'student', 'mobo'])
    parser.add_argument('--teacher_type', type=str, default='Inception',choices=['Inception', 'CIF', 'Forest', 
                                                                                 'Proximity', 'TDE', 'Rocket', 
                                                                                 'Matrix'])
    parser.add_argument('--bo_init', type=int, default=10)
    parser.add_argument('--bo_steps', type=int, default=10)
    parser.add_argument('--evaluation', type=str, default='aed-loo', choices=['lightts','classic','agree','cawpe',
                                                                        'reinforced','aed-loo','fp-ensemble',
                                                                        'fp-student','non-distilled-student'])
    parser.add_argument('--kd_temperature', type=float, default=5)
    parser.add_argument('--teachers', type=int, default=10)
    parser.add_argument('--w_ce', type=float, default=0.5, help='weight for cross entropy')
    parser.add_argument('--w_kl', type=float, default=-1, help='weight for KL')
    parser.add_argument('--explore_branches', type=int, default=1)
    parser.add_argument('--val_epochs', type=int, default=1)
    parser.add_argument('--gumbel', type=float, default=1.2)
    parser.add_argument('--cross_validation', type=int, default=5)
    
    config = parser.parse_args()
    
    if config.device == -1:
        config.device = torch.device(get_free_device())
    else:
        config.device = torch.device("cuda:" + str(config.device))
    
    if config.init_seed > -1:
        random.seed(config.init_seed)
        np.random.seed(config.init_seed)
        torch.manual_seed(config.init_seed)
        torch.cuda.manual_seed(config.init_seed)
        torch.backends.cudnn.deterministic = True

    if config.evaluation == 'fp-ensemble':
        config.experiment = 'teacher'

    df = pd.read_csv('TimeSeries.csv',header=None)
    num_classes = int(df[(df == config.dataset).any(axis=1)][1])
    if num_classes == 2:
        num_classes = 1
    config.num_classes = num_classes

    config.data_folder = pathlib.Path('./dataset/TimeSeriesClassification')
    
    if config.experiment == 'teacher':
        RunTeacher(config)
    elif config.experiment == 'student':
        RunStudent(config)
    elif config.experiment == 'mobo':
        print() #BayesianOptimization from bo.py
