'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')  # This should be src/lightgcn/data
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')

import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

config = {
    'all_models': ['mf', 'lgn'],
    'all_dataset': ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'music'],
    'bpr_batch': 2048,
    'recdim': 64,
    'layer': 3,
    'lr': 0.001,
    'decay': 1e-4,
    'dropout': 0,
    'keep_prob': 0.6,
    'a_fold': 100,
    'testbatch': 100,
    'dataset': 'music',
    'path': os.path.abspath(join(ROOT_PATH, 'data')),  # Point to src/lightgcn/data
    'topks': [20],
    'tensorboard': 1,
    'comment': 'lightgcn',
    'load': 0,
    'epochs': 1000,
    'multicore': 0,
    'pretrain': 0,
    'seed': 2020,
    'model': 'lgn',
    'batch_size': 4096,
    'bpr_batch_size': 2048,
    'latent_dim_rec': 64,
    'lightGCN_n_layers': 3,
    'keep_prob': 1.0,
    'A_n_fold': 100,
    'test_u_batch_size': 100,
    'A_split': False,
    'bigdata': False,
}

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = config['seed']

dataset = config['dataset']
model_name = config['model']

if dataset not in config['all_dataset']:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {config['all_dataset']}")
if model_name not in config['all_models']:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {config['all_models']}")

TRAIN_epochs = config['epochs']
LOAD = config['load']
PATH = config['path']
tensorboard = config['tensorboard']
topks = config['topks']
comment = config['comment']

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
