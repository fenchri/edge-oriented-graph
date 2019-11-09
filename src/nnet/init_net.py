#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21-Feb-2019

author: fenia
"""

import torch
from torch import nn
from nnet.modules import EmbedLayer, Encoder, Classifier
from nnet.attention import Dot_Attention
from nnet.walks import WalkLayer


class BaseNet(nn.Module):
    def __init__(self, params, pembeds, sizes=None, maps=None, lab2ign=None):
        super(BaseNet, self).__init__()

        self.edg = ['MM', 'SS', 'ME', 'MS', 'ES', 'EE']

        self.dims = {}
        for k in self.edg:
            self.dims[k] = 4 * params['lstm_dim']

        self.device = torch.device("cuda:{}".format(params['gpu']) if params['gpu'] != -1 else "cpu")

        self.encoder = Encoder(input_size=params['word_dim'],
                               rnn_size=params['out_dim'],
                               num_layers=1,
                               bidirectional=True,
                               dropout=0.0)

        self.word_embed = EmbedLayer(num_embeddings=sizes['word_size'],
                                     embedding_dim=params['word_dim'],
                                     dropout=params['drop_i'],
                                     ignore=None,
                                     freeze=params['freeze_words'],
                                     pretrained=pembeds,
                                     mapping=maps['word2idx'])

        if params['dist']:
            self.dims['MM'] += params['dist_dim']
            self.dims['SS'] += params['dist_dim']
            self.dist_embed = EmbedLayer(num_embeddings=sizes['dist_size'] + 1,
                                         embedding_dim=params['dist_dim'],
                                         dropout=0.0,
                                         ignore=sizes['dist_size'],
                                         freeze=False,
                                         pretrained=None,
                                         mapping=None)

        if params['context']:
            self.dims['MM'] += (2 * params['lstm_dim'])
            self.attention = Dot_Attention(input_size=2 * params['lstm_dim'],
                                           device=self.device,
                                           scale=False)

        if params['types']:
            for k in self.edg:
                self.dims[k] += (2 * params['type_dim'])

            self.type_embed = EmbedLayer(num_embeddings=3,
                                         embedding_dim=params['type_dim'],
                                         dropout=0.0,
                                         freeze=False,
                                         pretrained=None,
                                         mapping=None)

        self.reduce = nn.ModuleDict()
        for k in self.edg:
            if k != 'EE':
                self.reduce.update({k: nn.Linear(self.dims[k], params['out_dim'], bias=False)})
            elif (('EE' in params['edges']) or ('FULL' in params['edges'])) and (k == 'EE'):
                self.ee = True
                self.reduce.update({k: nn.Linear(self.dims[k], params['out_dim'], bias=False)})
            else:
                self.ee = False

        if params['walks_iter'] and params['walks_iter'] > 0:
            self.walk = WalkLayer(input_size=params['out_dim'],
                                  iters=params['walks_iter'],
                                  beta=params['beta'],
                                  device=self.device)

        self.classifier = Classifier(in_size=params['out_dim'],
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])
        self.loss = nn.CrossEntropyLoss()

        # hyper-parameters for tuning
        self.beta = params['beta']
        self.dist_dim = params['dist_dim']
        self.type_dim = params['type_dim']
        self.drop_i = params['drop_i']
        self.drop_o = params['drop_o']
        self.gradc = params['gc']
        self.learn = params['lr']
        self.reg = params['reg']
        self.out_dim = params['out_dim']

        # other parameters
        self.mappings = {'word': maps['word2idx'], 'type': maps['type2idx'], 'dist': maps['dist2idx']}
        self.inv_mappings = {'word': maps['idx2word'], 'type': maps['idx2type'], 'dist': maps['idx2dist']}
        self.word_dim = params['word_dim']
        self.lstm_dim = params['lstm_dim']
        self.walks_iter = params['walks_iter']
        self.rel_size = sizes['rel_size']
        self.types = params['types']
        self.ignore_label = lab2ign
        self.context = params['context']
        self.dist = params['dist']

