#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/02/19

author: fenia
"""

import sys
import os
from tabulate import tabulate
import itertools
import numpy as np
import pickle as pkl
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def solve(A, B):
    A = list(map(int, A))
    B = list(map(int, B))
    m = len(A)
    n = len(B)
    A.sort()
    B.sort()
    a = 0
    b = 0
    result = sys.maxsize

    while a < m and b < n:
        if abs(A[a] - B[b]) < result:
            result = abs(A[a] - B[b])

        # Move Smaller Value
        if A[a] < B[b]:
            a += 1
        else:
            b += 1
    # return final sma result
    return result


def write_errors(preds, info, ofile, map_=None):
    """ Write model errors to file """
    print('Saving predictions ... ', end="")
    with open(ofile+'.errors', 'w') as outfile:
        for p, i in zip(preds, info):
            i = [i_ for i_ in i if i_]
            assert len(p) == len(i)

            for k, j in zip(p, i):
                if k != j['rel']:
                    outfile.write('Prediction: {} \t Truth: {} \t Type: {} \n'.format(map_[k], map_[j['rel']], j['cross']))
                    doc = [it for items in j['doc'] for it in items]
                    outfile.write('{}\n{}\n'.format(j['pmid'], ' '.join(doc)))

                    gg1 = ' | '.join([' '.join(doc[int(m1):int(m2)]) for m1,m2 in
                                      zip(j['entA'].mstart.split(':'), j['entA'].mend.split(':'))])
                    gg2 = ' | '.join([' '.join(doc[int(m1):int(m2)]) for m1, m2 in
                                      zip(j['entB'].mstart.split(':'), j['entB'].mend.split(':'))])

                    outfile.write('Arg1: {} | {}\n'.format(j['entA'].id, gg1))
                    outfile.write('Arg2: {} | {}\n'.format(j['entB'].id, gg2))
                    outfile.write('Distance: {}\n'.format(solve(j['sentA'].split(':'), j['sentB'].split(':'))))
                    outfile.write('\n')
    print('DONE')


def write_preds(preds, info, ofile, map_=None):
    """ Write predictions to file """
    print('Saving errors ... ', end="")
    with open(ofile+'.preds', 'w') as outfile:
        for p, i in zip(preds, info):
            i = [i_ for i_ in i if i_]
            assert len(p) == len(i)

            for k, j in zip(p, i):
                # pmid, e1, e2, pred, truth
                if map_[k] == '1:NR:2':
                    pass
                else:
                    outfile.write('{}\n'.format('|'.join([j['pmid'].split('__')[0],
                                                          j['entA'].id, j['entB'].id, j['cross'],
                                                          str(solve(j['sentA'].split(':'), j['sentB'].split(':'))),
                                                          map_[k]])))
    print('DONE')


def plot_learning_curve(trainer, model_folder):
    """
    Plot the learning curves for training and test set (loss and primary score measure)

    Args:
        trainer (Class): trainer object
        model_folder (str): folder to save figures
    """
    x = list(map(int, np.arange(len(trainer.train_res['loss']))))
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, trainer.train_res['loss'], 'b', label='train')
    plt.plot(x, trainer.test_res['loss'], 'g', label='test')
    plt.legend()
    plt.ylabel('Loss')
    plt.yticks(np.arange(0, 1, 0.1))

    plt.subplot(2, 1, 2)
    plt.plot(x, trainer.train_res['score'], 'b', label='train')
    plt.plot(x, trainer.test_res['score'], 'g', label='test')
    plt.legend()
    plt.ylabel('F1-score')
    plt.xlabel('Epochs')
    plt.yticks(np.arange(0, 1, 0.1))

    fig.savefig(model_folder + '/learn_curves.png', bbox_inches='tight')


def print_results(scores, scores_class, show_class, time):
    """
    Print class-wise results.

    Args:
        scores (dict): micro and macro scores
        scores_class: score per class
        show_class (bool): show or not
        time: time
    """

    def indent(txt, spaces=18):
        return "\n".join(" " * spaces + ln for ln in txt.splitlines())

    if show_class:
        # print results for every class
        scores_class.append(['-----', None, None, None])
        scores_class.append(['macro score', scores['macro_p'], scores['macro_r'], scores['macro_f']])
        scores_class.append(['micro score', scores['micro_p'], scores['micro_r'], scores['micro_f']])
        print(' | {}\n'.format(humanized_time(time)))
        print(indent(tabulate(scores_class,
                              headers=['Class', 'P', 'R', 'F1'],
                              tablefmt='orgtbl',
                              floatfmtL=".4f",
                              missingval="")))
        print()
    else:
        print('ACC = {:.04f} , '
              'MICRO P/R/F1 = {:.04f}\t{:.04f}\t{:.04f} | '.format(scores['acc'], scores['micro_p'], scores['micro_r'],
                                                                   scores['micro_f']), end="")

        l = ':<7'  # +str(len(str(scores['total'])))
        s = 'TP/ACTUAL/PRED = {'+l+'}/{'+l+'}/{'+l+'}, TOTAL {'+l+'}'
        print(s.format(scores['tp'], scores['true'], scores['pred'], scores['total']), end="")
        print(' | {}'.format(humanized_time(time)))


class Tee(object):
    """
    Object to print stdout to a file.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f_ in self.files:
            f_.write(obj)
            f_.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f_ in self.files:
            f_.flush()


def humanized_time(second):
    """
    :param second: time in seconds
    :return: human readable time (hours, minutes, seconds)
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def setup_log(params, mode):
    """
    Setup .log file to record training process and results.

    Args:
        params (dict): model parameters

    Returns:
        model_folder (str): model directory
    """
    if params['walks_iter'] == 0:
        length = 1
    elif not params['walks_iter']:
        length = 1
    else:
        length = 2**params['walks_iter']
    # folder_name = 'b{}-wd{}-ld{}-od{}-td{}-beta{}-pd{}-di{}-do{}-lr{}-gc{}-r{}-p{}-walks{}'.format(
    #     params['batch'], params['word_dim'], params['lstm_dim'], params['out_dim'], params['type_dim'],
    #     params['pos_dim'], params['beta'], params['pos_dim'], params['drop_i'], params['drop_o'], params['lr'],
    #     params['gc'], params['reg'], params['patience'], length)

    folder_name = 'b{}-walks{}'.format(params['batch'], length)

    folder_name += '_'+'_'.join(params['edges'])

    if params['context']:
        folder_name += '_context'

    if params['types']:
        folder_name += '_types'

    if params['dist']:
        folder_name += '_dist'

    if params['freeze_words']:
        folder_name += '_freeze'

    if params['window']:
        folder_name += '_win'+str(params['window'])

    model_folder = params['folder'] + '/' + folder_name
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    log_file = model_folder + '/info_'+mode+'.log'

    f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, f)
    return model_folder


def observe(model):
    """
    Observe model parameters: name, range of matrices & gradients

    Args
       model: specified model object
    """
    for name, param in model.named_parameters():
        p_data, p_grad = param.data, param.grad.data
        print('Name: {:<30}\tRange of data: [{:.4f}, {:.4f}]\tRange of gradient: [{:.4f}, {:.4f}]'.format(name,
              np.min(p_data.data.to('cpu').numpy()),
              np.max(p_data.data.to('cpu').numpy()),
              np.min(p_grad.data.to('cpu').numpy()),
              np.max(p_grad.data.to('cpu').numpy())))
    print('--------------------------------------')


def save_model(model_folder, trainer, loader):
    print('\nSaving the model & the parameters ...')
    # save mappings
    with open(os.path.join(model_folder, 'mappings.pkl'), 'wb') as f:
        pkl.dump(loader, f, pkl.HIGHEST_PROTOCOL)
    torch.save(trainer.model.state_dict(), os.path.join(model_folder, 're.model'))


def load_model(model_folder, trainer):
    print('\nLoading model & parameters ...')
    trainer.model.load_state_dict(torch.load(os.path.join(model_folder, 're.model'),
                                             map_location=trainer.model.device))
    return trainer


def load_mappings(model_folder):
    with open(os.path.join(model_folder, 'mappings.pkl'), 'rb') as f:
        loader = pkl.load(f)
    return loader


def print_options(params):
    print('''\nParameters:
            - Train Data        {}
            - Test Data         {}
            - Embeddings        {}, Freeze: {}
            - Save folder       {}

            - batchsize         {}
            - Walks iteration   {} -> Length = {}
            - beta              {}

            - Context           {}
            - Node Type         {}
            - Distances         {}
            - Edge Types        {}
            - Window            {}
            
            - Epoch             {}
            - UNK word prob     {}
            - Parameter Average {}
            - Early stop        {} -> Patience = {}
            - Regularization    {}
            - Gradient Clip     {}
            - Dropout I/O       {}/{}
            - Learning rate     {}
            - Seed              {}
            '''.format(params['train_data'], params['test_data'], params['embeds'], params['freeze_words'],
                       params['folder'],
                       params['batch'],
                       params['walks_iter'], 2 ** params['walks_iter'] if params['walks_iter'] else 0, params['beta'],
                       params['context'], params['types'], params['dist'], params['edges'],
                       params['window'], params['epoch'],
                       params['unk_w_prob'], params['param_avg'],
                       params['early_stop'], params['patience'],
                       params['reg'], params['gc'], params['drop_i'], params['drop_o'], params['lr'], params['seed']))
