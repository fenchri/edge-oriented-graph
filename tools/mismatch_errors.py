#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/08/2019

author: fenia
"""

import os, re, sys
import numpy as np
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--errorA', type=str)
parser.add_argument('--errorB', type=str)
parser.add_argument('--truth', type=str)
parser.add_argument('--type', type=str, help='CROSS or NON-CROSS')
args = parser.parse_args()

data = OrderedDict()
for files in [args.errorA, args.errorB]:
    data[files] = []
    with open(files) as a:
        lines = a.readlines()
        for i, line in enumerate(lines):
            if line.startswith('Prediction:'):
                line = line.rstrip().split('\t')

                pred = line[0].split('Prediction: ')[1].rstrip()
                truth = line[1].split('Truth: ')[1].rstrip()
                type_ = line[2].split('Type: ')[1].rstrip()
                pmid = lines[i+1].rstrip()
                distance = int(lines[i + 5].rstrip().split(' ')[1])
                no_arg1 = len(lines[i + 3].rstrip().split(' | ')) - 1
                no_arg2 = len(lines[i + 4].rstrip().split(' | ')) - 1
                arg1 = lines[i + 3].rstrip().split(' | ')[0].strip('Arg1: ')
                arg2 = lines[i + 4].rstrip().split(' | ')[0].strip('Arg2: ')

                data[files] += [OrderedDict({'pmid': pmid, 'pred': pred, 'truth': truth, 'dist': distance,
                                             'no1': no_arg1, 'no2': no_arg2, 'arg1': arg1, 'arg2': arg2,
                                             'type': type_})]


data['truth'] = []
with open(args.truth) as tr:
    for line in tr:
        line = line.rstrip().split('|')
        data['truth'] += [OrderedDict({'pmid': line[0], 'arg1': line[1], 'arg2': line[2], 'type': line[3],
                                       'dist': line[4]})]


not_found_in_b = []
common_fp, common_fn, common_tp = [], [], []
fns_a, fns_b = [], []
fps_a, fps_b = [], []
tps_a, tps_b = [], []

for a in data[args.errorA]:
    cnt = 0
    for b in data[args.errorB]:
        if a['pmid'] == b['pmid'] and a['arg1'] == b['arg1'] and a['arg2'] == b['arg2'] and a['type'] == b['type'] and \
                a['pred'] == b['pred']:  # found
            cnt += 1
            break

    if cnt == 0 and a['type'] == args.type and a['pred'] == '1:NR:2':
        fns_a += [a]  # element of a does not exist in b
    elif cnt == 0 and a['type'] == args.type and a['pred'] != '1:NR:2':
        fps_a += [a]
    elif cnt == 1 and a['type'] == args.type and a['pred'] == '1:NR:2':
        common_fn += [a]
    elif cnt == 1 and a['type'] == args.type and a['pred'] != '1:NR:2':
        common_fp += [a]


for a in data[args.errorB]:
    cnt = 0
    for b in data[args.errorA]:
        if a['pmid'] == b['pmid'] and a['arg1'] == b['arg1'] and a['arg2'] == b['arg2'] and a['type'] == b['type'] and \
                a['pred'] == b['pred']:  # found
            cnt += 1
            break

    if cnt == 0 and a['type'] == args.type and a['pred'] == '1:NR:2':
        fns_b += [a]  # element of a does not exist in b
    elif cnt == 0 and a['type'] == args.type and a['pred'] != '1:NR:2':
        fps_b += [a]


for t in data['truth']:
    cnta = 0
    for a in data[args.errorA]:
        if a['pmid'] == t['pmid'] and a['arg1'] == t['arg1'] and a['arg2'] == t['arg2']:
            cnta += 1

    cntb = 0
    for b in data[args.errorB]:
        if b['pmid'] == t['pmid'] and b['arg1'] == t['arg1'] and b['arg2'] == t['arg2']:
            cntb += 1

    if cnta == 0 and cntb == 0 and t['type'] == args.type:
        common_tp += [a]
    elif cnta == 0 and t['type'] == args.type:
        tps_a += [a]
    elif cntb == 0 and t['type'] == args.type:
        tps_b += [b]


print('   TP    FP    FN   \n',
      '-------------------\n'
      'A: {:<5} {:<5} {:<5}\n'
      'B: {:<5} {:<5} {:<5}\n'
      'Common FP: {}\n'
      'Common FN: {}\n'
      'Common TP: {} / {}'.format(len(tps_a), len(fps_a), len(fns_a),
                                  len(tps_b), len(fps_b), len(fns_b),
                                  len(common_fp), len(common_fn), len(common_tp),
                                  len([d for d in data['truth'] if d['type'] == args.type])))

print()

print('Common FNs')
for nf in common_fn:
    print(nf)






