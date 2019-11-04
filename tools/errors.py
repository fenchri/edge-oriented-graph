#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/08/2019

author: fenia
"""

import os, re, sys
import numpy as np

data = []
with open(sys.argv[1], 'r') as infile:
    lines = infile.readlines()
    for i, line in enumerate(lines):
        if line.startswith('Prediction'):
            line = line.rstrip().split('\t')
            pred = line[0].split('Prediction: ')[1].rstrip()
            truth = line[1].split('Truth: ')[1].rstrip()
            distance = int(lines[i+5].rstrip().split(' ')[1])
            no_arg1 = len(lines[i+3].rstrip().split(' | ')) - 1
            no_arg2 = len(lines[i+4].rstrip().split(' | ')) - 1

            data += [{'pred': pred, 'truth': truth, 'dist': distance, 'no1': no_arg1, 'no2': no_arg2}]

# number of FPs
fps = 0
fns = 0
fps_dist = 0
fns_dist = 0
fps_dist_1 = 0
fns_dist_1 = 0

arg1 = 0
both_args = 0

for d in data:
    if d['pred'] == '1:CID:2' and d['truth'] == '1:NR:2' and d['dist'] > 0:
        fps += 1

    if d['pred'] == '1:NR:2' and d['truth'] == '1:CID:2' and d['dist'] > 0:
        fns += 1

    if d['pred'] == '1:CID:2' and d['truth'] == '1:NR:2' and d['dist'] > 1:
        fps_dist += 1

    if d['pred'] == '1:NR:2' and d['truth'] == '1:CID:2' and d['dist'] > 1:
        fns_dist += 1

    if d['pred'] == '1:CID:2' and d['truth'] == '1:NR:2' and d['dist'] == 1:
        fps_dist_1 += 1

    if d['pred'] == '1:NR:2' and d['truth'] == '1:CID:2' and d['dist'] == 1:
        fns_dist_1 += 1

    if d['pred'] == '1:NR:2' and d['truth'] == '1:CID:2' and (d['no1'] > 1 or d['no2'] > 1) and d['dist'] >= 1:
        arg1 += 1

    if d['pred'] == '1:NR:2' and d['truth'] == '1:CID:2' and (d['no1'] == 1 and d['no2'] == 1) and d['dist'] >= 1:
        both_args += 1


print('[Inter] FPs:', fps)
print('[Inter] FNs:', fns)
print('[Inter] FPs & dist > 1:', fps_dist)
print('[Inter] FNs & dist > 1:', fns_dist)
print('[Inter] FPs & dist = 1:', fps_dist_1)
print('[Inter] FNs & dist = 1:', fns_dist_1)

print('[FN, Inter] Both args > 1 mention:', arg1)
#print('[FN, Inter] Both arguments with 1 mention:', both_args)

