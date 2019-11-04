#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/09/2019

author: fenia
"""

import os, re, sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', '-i', type=str)
parser.add_argument('--output_file', '-o', type=str)
args = parser.parse_args()

if not os.path.exists('/'.join(args.output_file.split('/')[:-1])):
    os.makedirs('/'.join(args.output_file.split('/')[:-1]))

abstracts = {}
entities = {}
relations = {}
with open(args.input_folder + 'abstracts.txt', 'r') as infile:
    for line in infile:
        if line.rstrip().isdigit():
            pmid = line.rstrip()
            abstracts[pmid] = []
            relations[pmid] = []
            entities[pmid] = []

        elif line != '\n':
            abstracts[pmid] += [line.rstrip()]

with open(args.input_folder + 'anns.txt', 'r') as infile:
    for line in infile:
        line = line.split('\t')

        if line[0].isdigit():
            entities[line[0]] += [tuple(line)]

with open(args.input_folder + 'labels.csv', 'r') as infile:
    for line in infile:
        line = line.split(',')

        if line[0].isdigit() and line[3].rstrip() == '1':
            line = ','.join(line).rstrip().split(',')

            relations[line[0]] += [tuple([line[0]] + ['GDA'] + line[1:-1])]

with open(args.output_file, 'w') as outfile:
    for d in tqdm(abstracts.keys(), desc='Writing 2 PubTator format'):
        if len(abstracts[d]) > 2:
            print('something is wrong')
            exit(0)

        for i in range(0, len(abstracts[d])):
            if i == 0:
                outfile.write('{}|t|{}\n'.format(d, abstracts[d][i]))
            else:
                outfile.write('{}|a|{}\n'.format(d, abstracts[d][i]))

        for e in entities[d]:
            outfile.write('{}'.format('\t'.join(e)))

        for r in relations[d]:
            outfile.write('{}\n'.format('\t'.join(r)))
        outfile.write('\n')

