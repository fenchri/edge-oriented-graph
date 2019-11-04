#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21-Feb-2019

author: fenia
"""

import torch
import random
import numpy as np
from dataset import DocRelationDataset
from loader import DataLoader, ConfigLoader
from nnet.trainer import Trainer
from utils import setup_log, save_model, load_model, plot_learning_curve, load_mappings


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(parameters):
    model_folder = setup_log(parameters, 'train')

    set_seed(parameters['seed'])

    ###################################
    # Data Loading
    ###################################
    print('Loading training data ...')
    train_loader = DataLoader(parameters['train_data'], parameters)
    train_loader(embeds=parameters['embeds'])
    train_data = DocRelationDataset(train_loader, 'train', parameters, train_loader).__call__()

    print('\nLoading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters)
    test_loader()
    test_data = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()

    ###################################
    # Training
    ###################################
    trainer = Trainer(train_loader, parameters, {'train': train_data, 'test': test_data}, model_folder)
    trainer.run()

    if parameters['plot']:
        plot_learning_curve(trainer, model_folder)

    if parameters['save_model']:
        save_model(model_folder, trainer, train_loader)


def test(parameters):
    model_folder = setup_log(parameters, 'test')

    print('\nLoading mappings ...')
    train_loader = load_mappings(model_folder)
    
    print('\nLoading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters)
    test_loader()    
    test_data = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__() 

    m = Trainer(train_loader, parameters, {'train': [], 'test': test_data}, model_folder)
    trainer = load_model(model_folder, m)
    trainer.eval_epoch(final=True, save_predictions=True)


def main():
    config = ConfigLoader()
    parameters = config.load_config()

    if parameters['train']:
        train(parameters)

    elif parameters['test']:
        test(parameters)


if __name__ == "__main__":
    main()

