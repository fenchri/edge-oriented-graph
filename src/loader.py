#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 21-Feb-2019

author: fenia
"""

import numpy as np
import argparse
import yaml
import yamlordereddictloader
from collections import OrderedDict
from reader import read, read_subdocs


def str2bool(i):
    if isinstance(i, bool):
        return i
    if i.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif i.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ConfigLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_cmd():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='Yaml parameter file')
        parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
        parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
        parser.add_argument('--gpu', type=int, help='GPU number')
        parser.add_argument('--walks', type=int, help='Number of walk iterations')
        parser.add_argument('--window', type=int, help='Window for training (empty processes the whole document, '
                                                       '1 processes 1 sentence at a time, etc)')
        parser.add_argument('--edges', nargs='*', help='Edge types')
        parser.add_argument('--types', type=str2bool, help='Include node types (Boolean)')
        parser.add_argument('--context', type=str2bool, help='Include MM context (Boolean)')
        parser.add_argument('--dist', type=str2bool, help='Include distance (Boolean)')
        parser.add_argument('--example', help='Show example', action='store_true')
        parser.add_argument('--seed', help='Fixed random seed number', type=int)
        parser.add_argument('--early_stop', action='store_true', help='Use early stopping')
        parser.add_argument('--epoch', type=int, help='Maximum training epoch')
        return parser.parse_args()

    def load_config(self):
        inp = self.load_cmd()
        with open(vars(inp)['config'], 'r') as f:
            parameters = yaml.load(f, Loader=yamlordereddictloader.Loader)

        parameters = dict(parameters)
        if not inp.train and not inp.test:
            print('Please specify train/test mode.')
            sys.exit(0)

        parameters['train'] = inp.train
        parameters['test'] = inp.test
        parameters['gpu'] = inp.gpu
        parameters['example'] = inp.example

        if inp.walks and inp.walks >= 0:
            parameters['walks_iter'] = inp.walks

        if inp.edges:
            parameters['edges'] = inp.edges

        if inp.types != None:
            parameters['types'] = inp.types
        
        if inp.dist != None:
            parameters['dist'] = inp.dist
        
        if inp.window:
            parameters['window'] = inp.window

        if inp.context != None:
            parameters['context'] = inp.context
       
        if inp.seed:
            parameters['seed'] = inp.seed

        if inp.epoch:
            parameters['epoch'] = inp.epoch

        if inp.early_stop:
            parameters['early_stop'] = True

        return parameters


class DataLoader:
    def __init__(self, input_file, params):
        self.input = input_file
        self.params = params

        self.pre_words = []
        self.pre_embeds = OrderedDict()
        self.max_distance = -9999999999
        self.singletons = []
        self.label2ignore = -1
        self.ign_label = self.params['label2ignore']

        self.word2index, self.index2word, self.n_words, self.word2count = {'<UNK>': 0}, {0: '<UNK>'}, 1, {'<UNK>': 1}
        self.type2index, self.index2type, self.n_type, self.type2count = {'<ENT>': 0, '<MENT>': 1, '<SENT>': 2}, \
                                                                         {0: '<ENT>', 1: '<MENT>', 2: '<SENT>'}, 3, \
                                                                         {'<ENT>': 1, '<MENT>': 1, '<SENT>': 1}
        self.rel2index, self.index2rel, self.n_rel, self.rel2count = {}, {}, 0, {}
        self.dist2index, self.index2dist, self.n_dist, self.dist2count = {}, {}, 0, {}
        self.documents, self.entities, self.pairs = OrderedDict(), OrderedDict(), OrderedDict()

    def find_ignore_label(self):
        """
        Find relation Id to ignore
        """
        for key, val in self.index2rel.items():
            if val == self.ign_label:
                self.label2ignore = key
        assert self.label2ignore != -1

    @staticmethod
    def check_nested(p):
        starts1 = list(map(int, p[8].split(':')))
        ends1 = list(map(int, p[9].split(':')))

        starts2 = list(map(int, p[14].split(':')))
        ends2 = list(map(int, p[15].split(':')))

        for s1, e1, s2, e2 in zip(starts1, ends1, starts2, ends2):
            if bool(set(np.arange(s1, e1)) & set(np.arange(s2, e2))):
                print('nested pair', p)

    def find_singletons(self, min_w_freq=1):
        """
        Find items with frequency <= 2 and based on probability
        """
        self.singletons = frozenset([elem for elem, val in self.word2count.items()
                                     if (val <= min_w_freq) and elem != '<UNK>'])

    def add_relation(self, rel):
        if rel not in self.rel2index:
            self.rel2index[rel] = self.n_rel
            self.rel2count[rel] = 1
            self.index2rel[self.n_rel] = rel
            self.n_rel += 1
        else:
            self.rel2count[rel] += 1

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_type(self, type):
        if type not in self.type2index:
            self.type2index[type] = self.n_type
            self.type2count[type] = 1
            self.index2type[self.n_type] = type
            self.n_type += 1
        else:
            self.type2count[type] += 1

    def add_dist(self, dist):
        if dist not in self.dist2index:
            self.dist2index[dist] = self.n_dist
            self.dist2count[dist] = 1
            self.index2dist[self.n_dist] = dist
            self.n_dist += 1
        else:
            self.dist2count[dist] += 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_document(self, document):
        for sentence in document:
            self.add_sentence(sentence)

    def load_embeds(self, word_dim):
        """
        Load pre-trained word embeddings if specified
        """
        self.pre_embeds = OrderedDict()
        with open(self.params['embeds'], 'r') as vectors:
            for x, line in enumerate(vectors):

                if x == 0 and len(line.split()) == 2:
                    words, num = map(int, line.rstrip().split())
                else:
                    word = line.rstrip().split()[0]
                    vec = line.rstrip().split()[1:]

                    n = len(vec)
                    if n != word_dim:
                        print('Wrong dimensionality! -- line No{}, word: {}, len {}'.format(x, word, n))
                        continue
                    self.add_word(word)
                    self.pre_embeds[word] = np.asarray(vec, 'f')
        self.pre_words = [w for w, e in self.pre_embeds.items()]
        print('  Found pre-trained word embeddings: {} x {}'.format(len(self.pre_embeds), word_dim), end="")

    def find_max_length(self, lengths):
        self.max_distance = max(lengths) - 1

    def read_n_map(self):
        """
        Read input.
        Lengths is the max distance for each document
        """
        if not self.params['window']:
            lengths, sents, self.documents, self.entities, self.pairs = \
                read(self.input, self.documents, self.entities, self.pairs)
        else:
            lengths, sents, self.documents, self.entities, self.pairs = \
                read_subdocs(self.input, self.params['window'], self.documents, self.entities, self.pairs)

        self.find_max_length(lengths)

        # map types and positions and relation types
        for did, d in self.documents.items():
            self.add_document(d)

        for did, e in self.entities.items():
            for k, v in e.items():
                self.add_type(v.type)

        for dist in np.arange(0, self.max_distance+1):
            self.add_dist(dist)

        for did, p in self.pairs.items():
            for k, v in p.items():
                if v.type == 'not_include':
                    continue
                self.add_relation(v.type)
        assert len(self.entities) == len(self.documents) == len(self.pairs)

    def statistics(self):
        """
        Print statistics for the dataset
        """
        print('  Documents: {:<5}\n  Words: {:<5}'.format(len(self.documents), self.n_words))

        print('  Relations: {}'.format(sum([v for k, v in self.rel2count.items()])))
        for k, v in sorted(self.rel2count.items()):
            print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.rel2index[k]))

        print('  Entities: {}'.format(sum([len(e) for e in self.entities.values()])))
        for k, v in sorted(self.type2count.items()):
            print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.type2index[k]))

        print('  Singletons: {}/{}'.format(len(self.singletons), self.n_words))

    def __call__(self, embeds=None):
        self.read_n_map()
        self.find_ignore_label()
        self.find_singletons(self.params['min_w_freq'])  # words with freq=1
        self.statistics()
        if embeds:
            self.load_embeds(self.params['word_dim'])
            print(' --> Words + Pre-trained: {:<5}'.format(self.n_words))
