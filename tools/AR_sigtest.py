peventmine
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/08/2019

author: fenia
"""

import argparse
from collections import OrderedDict
from tqdm import tqdm
import random
import numpy as np

map_ = {'1:CID:2': 1, '1:NR:2': 0}


def load_data(args):
    all, intra, inter = {}, {}, {}
    all['A'], all['B'], all['true'] = {}, {}, {}
    intra['A'], intra['B'], intra['true'] = {}, {}, {}
    inter['A'], inter['B'], inter['true'] = {}, {}, {}

    for system, typ in zip([args.systemA, args.systemB, args.truth], ['A', 'B', 'true']):
        with open(system, 'r') as pred:
            for line in pred:
                line = line.rstrip().split('|')

                # format: {(PMID, arg1, arg2): rlabel}
                if (line[0], line[1], line[2], line[3], line[4]) not in all[typ]:
                    all[typ].update({(line[0], line[1], line[2]): map_[line[4]]})

                if ((line[0], line[1], line[2], line[4]) not in inter[typ]) and (line[3] == 'CROSS'):
                    inter[typ].update({(line[0], line[1], line[2]): map_[line[4]]})

                if ((line[0], line[1], line[2], line[4]) not in intra[typ]) and (line[3] == 'NON-CROSS'):
                    intra[typ].update({(line[0], line[1], line[2]): map_[line[4]]})

    return all, intra, inter


def align(all, intra, inter):
    all_n = {'A': OrderedDict(), 'B': OrderedDict(), 'true': OrderedDict()}
    intra_n = {'A': OrderedDict(), 'B': OrderedDict(), 'true': OrderedDict()}
    inter_n = {'A': OrderedDict(), 'B': OrderedDict(), 'true': OrderedDict()}

    for typ, typ_n in zip([all, intra, inter], [all_n, intra_n, inter_n]):
        for key in list(typ['A'].keys()) + list(typ['B'].keys()) + list(typ['true'].keys()):
            if key in typ['A']:
                typ_n['A'][key] = typ['A'][key]
            else:
                typ_n['A'][key] = 0

            if key in typ['B']:
                typ_n['B'][key] = typ['B'][key]
            else:
                typ_n['B'][key] = 0

            if key in typ['true']:
                typ_n['true'][key] = typ['true'][key]
            else:
                typ_n['true'][key] = 0

    return all_n, intra_n, inter_n


def eval_(t, y):
    t = list(t.values())
    y = list(y.values())

    label_num = 2
    ignore_label = 0

    mask_t = np.equal(t, ignore_label)  # where the ground truth needs to be ignored
    mask_p = np.equal(y, ignore_label)  # where the predicted needs to be ignored

    true = np.where(mask_t, label_num, t)  # ground truth
    pred = np.where(mask_p, label_num, y)  # output of NN

    tp_mask = np.where(np.equal(pred, true), true, label_num)
    fp_mask = np.where(np.not_equal(pred, true), pred, label_num)
    fn_mask = np.where(np.not_equal(pred, true), true, label_num)

    tp = np.sum(np.bincount(tp_mask, minlength=label_num)[:label_num])
    fp = np.sum(np.bincount(fp_mask, minlength=label_num)[:label_num])
    fn = np.sum(np.bincount(fn_mask, minlength=label_num)[:label_num])
    return prf(tp, fp, fn)


def prf(tp, fp, fn):
    micro_p = float(tp) / (tp + fp) if (tp + fp != 0) else 0.0
    micro_r = float(tp) / (tp + fn) if (tp + fn != 0) else 0.0
    micro_f = ((2 * micro_p * micro_r) / (micro_p + micro_r)) if micro_p != 0.0 and micro_r != 0.0 else 0.0
    # return micro_p, micro_r, micro_f
    return micro_f


def sig_test(args, system_A, system_B, truth):
    """
    Approximate Randomization significance test
    https://cs.stanford.edu/people/wmorgan/sigtest.pdf
    """
    r = 0
    for R_ in tqdm(range(0, args.R)):
        listX = OrderedDict()
        listY = OrderedDict()
        k = 0

        for d in system_A.keys():
            choose = random.randint(0, 1)
            if choose == 0:
                listX[d] = system_A[d]
                listY[d] = system_B[d]
            else:
                listX[d] = system_B[d]
                listY[d] = system_A[d]

        t_xy = np.abs(eval_(listX, truth) - eval_(listY, truth))
        t_ab = np.abs(eval_(system_A, truth) - eval_(system_B, truth))

        if t_xy >= t_ab:
            r += 1

    significance = (r+1)/(args.R+1)
    if significance < 0.05:
        decision = 'SIG !!! :D'
    else:
        decision = 'NOT SIG :('
    print('Significance: {} ==> {}'.format(significance, decision))
    print('========================')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systemA', type=str, help='predictions for system A')
    parser.add_argument('--systemB', type=str, help='predictions for system B')
    parser.add_argument('--truth', type=str, help='true values')
    parser.add_argument('--R', type=int, default=10000)
    args = parser.parse_args()

    a_, ia_, ie_ = load_data(args)
    a_2, ia_2, ie_2 = align(a_, ia_, ie_)

    print('=== OVERALL ===')
    print('System A:', eval_(a_2['A'], a_2['true']))
    print('System B:', eval_(a_2['B'], a_2['true']))
    sig_test(args, a_2['A'], a_2['B'], a_2['true'])

    # -------
    print('=== INTRA ===')
    print('System A:', eval_(ia_2['A'], ia_2['true']))
    print('System B:', eval_(ia_2['B'], ia_2['true']))
    sig_test(args, ia_2['A'], ia_2['B'], ia_2['true'])

    # -------
    print('=== INTER ===')
    print('System A:', eval_(ie_2['A'], ie_2['true']))
    print('System B:', eval_(ie_2['B'], ie_2['true']))
    sig_test(args, ie_2['A'], ie_2['B'], ie_2['true'])


if __name__ == "__main__":
    main()