#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/04/2019

author: fenia
"""

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--suffix', type=str)
args = parser.parse_args()

folder = args.folder
if not args.suffix:
    exp = ''
else:
    exp = args.suffix

edge_map = {"MM_ME_MS_ES_SS-ind": 'EOG+SSind',
            "MM_ME_MS_ES_SS": 'EOG+SS',
            "MM_ME_MS_ES": '-SS',
            "MM_ME_MS_SS-ind": '-ES',
            "MM_ME_ES_SS-ind": '-MS',
            "MM_MS_ES_SS-ind": '-ME',
            "ME_MS_ES_SS-ind": '-MM',
            "ES_SS-ind": '-M nodes',
            "MM_ME": '-S nodes',
            "FULL": 'fully connected', 
            "EE": 'EE only'}

edges = ["MM_ME_MS_ES_SS-ind",
         "MM_ME_MS_ES_SS",
         "MM_ME_MS_ES",
         "MM_ME_MS_SS-ind",
         "MM_ME_ES_SS-ind",
         "MM_MS_ES_SS-ind",
         "ME_MS_ES_SS-ind",
         "ES_SS-ind",
         "MM_ME",
         "FULL",
         "EE"]


scores = {}
for e in edges:
    scores[e] = {}
    for w in ['1', '2', '4', '8', '16', '32']:
        scores[e][w] = {}

for edg in edges:
    for w in ['1', '2', '4', '8', '16', '32']:
        if os.path.exists(folder+'b2-walks'+w+'_'+edg+exp+'/cdr-dev.preds'):
            
            output = subprocess.check_output(["python3", "evaluate.py",
                                              "--pred", folder+'b2-walks'+w+'_'+edg+exp+'/cdr-dev.preds',
                                              "--gold", "../../../../RELATION_DATA/CDR-BioCreativeV/CDR_DEV.gold"])

        elif os.path.exists(folder+'b2-walks'+w+'_'+edg+exp+'/cdr.preds'):
            
            output = subprocess.check_output(["python3", "evaluate.py",
                                              "--pred", folder+'b2-walks'+w+'_'+edg+exp+'/cdr.preds',
                                              "--gold", "../../../../RELATION_DATA/CDR-BioCreativeV/CDR_DEV.gold"])

        else:
            continue

        output = output.decode("utf-8").rstrip().split('\n')
        scores[edg][w]['overall'] = output[0].split('\t')[1:4]
        scores[edg][w]['intra'] = output[1].split('\t')[1:4]
        scores[edg][w]['inter'] = output[2].split('\t')[1:4]

        with open(folder+'b2-walks'+w+'_'+edg+exp+'/info_train.log', 'r') as infile:
            lines = infile.readlines()
            for i, line in enumerate(lines):
                line = line.strip().split(' ')
                if line[0] == 'Best':
                    fin = lines[i+1]
                    fin = fin.strip().split(' ')

                    assert scores[edg][w]['overall'] == fin[13].split('\t'), '{} <> {}'.format(scores[edg][w],
                                                                                               fin[13])


for e in edges:
    new_edge = True
    if e in scores:
        for w in ['1', '2', '4', '8', '16', '32']:
            if (w in scores[e]) and (scores[e][w]):
                if new_edge:
                    print('{:<20}\t{:<5}\t{:<5}\t{}'.format(edge_map[e], w, 'overall', '\t'.join(scores[e][w]['overall'])))
                else:
                    print('{:<20}\t{:<5}\t{:<5}\t{}'.format('', w, 'overall', '\t'.join(scores[e][w]['overall'])))
                print('{:<20}\t{:<5}\t{:<5}\t{}'.format('', '', 'intra', '\t'.join(scores[e][w]['intra'])))
                print('{:<20}\t{:<5}\t{:<5}\t{}'.format('', '', 'inter', '\t'.join(scores[e][w]['inter'])))
                print()
                new_edge = False
        print()
