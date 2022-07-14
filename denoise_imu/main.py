import os
import sys
import torch
import learning
import networks
import dataset
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))

configs = {
    'net_params':{
        'channel': [6, 16, 32, 64, 128, 3],
        'dropout': 0.1,
        'kernel': [7, 7, 7, 7, 1],
        'dilation': [1, 4, 16, 64, 1],
        'momentum': 0.1,
    },

    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,
        'weight_decay': 1e-1,
        'amsgrad': False,
    },

    'loss_class': networks.GyroLoss,
    'loss': {
        'min_N': 4, # int(np.log2(min_train_freq))
        'max_N': 5, # int(np.log2(max_train_freq))
        'w':  1e6,
        'target': 'rotation matrix',
        'huber': 0.005,
        'dt': 0.01,
    },

    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 600,
        'T_mult': 2,
        'eta_min': 1e-3,
    },


    'dataloader': {
        'batch_size': 10,
        'shuffle': False,
    },
    'freq_val': 600,
    'n_epochs': 1800,
    'result_dir': os.path.join(base_dir, "result/"),
    'weights_path': os.path.join(base_dir, "result/", "weights.pt"),

    'data_dir': os.path.join(base_dir, 'data/'),
    'train_seqs': [
        '11024_20190730_091616_0',
        '11024_20190730_091616_1',
        '11024_20190730_091616_2',
        '11024_20190730_100503_0',
        '11024_20190730_100503_1',
        '11024_20190730_100503_2',
        ],
    'val_seqs': [
        '11024_20190730_091616_2'
        ],
    'test_seqs': [
        '11024_20190730_102927_0'
        ],
    'N': 32 * 500,
    'min_train_freq': 16,
    'max_train_freq': 32,
}

def main():
    learning_process = learning.GyroLearningBasedProcessing(configs)
    learning_process.train()
    learning_process.test()

if __name__ == "__main__":
  main()