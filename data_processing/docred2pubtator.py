#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/08/2019

author: fenia
"""

import argparse
import os, re, sys
import json
import itertools
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', '-i', type=str)
parser.add_argument('--output_file', '-o', type=str)
args = parser.parse_args()

cnt = 0
docs = OrderedDict()
ents = OrderedDict()
rels = OrderedDict()
with open(args.input_file) as infile:
    data = json.load(infile)

    for d in data:
        docs[d['title']] = ' '.join([' '.join(s) for s in d['sents']])

        all_sents = d['sents']

        ents[d['title']] = []
        for elem in d['vertexSet']:
            e_id = 'E_'+str(cnt)
            for mention in elem:
                off_list = list(itertools.chain.from_iterable(all_sents[:int(mention['sent_id'])]))
                off = ' '.join(off_list)

                if mention['sent_id'] == 0:
                    offsetx = len(off) + len(' '.join(all_sents[mention['sent_id']][:mention['pos'][0]]))
                else:
                    offsetx = len(off) + 1 + len(' '.join(all_sents[mention['sent_id']][:mention['pos'][0]])) + 1

                offsety = offsetx + len(' '.join(all_sents[mention['sent_id']][mention['pos'][0]:mention['pos'][1]]))

                ents[d['title']] += [(mention['name'], mention['type'], offsetx, offsety, e_id)]
            cnt += 1

        rels[d['title']] = []
        for elem in d['labels']:
            rels[d['title']] += [(elem['r'], 'E_'+str(elem['h']), 'E_'+str(elem['t']))]


with open(args.output_file, 'w') as outfile:
    for title in docs.keys():
        outfile.write('{}|a|{}\n'.format(title, docs[title]))

        for e in ents[title]:
            outfile.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(title, e[2], e[3], e[0], e[1], e[4]))

        for r in rels[title]:
            outfile.write('{}\t{}\t{}\t{}\n'.format(title, r[0], r[1], r[2]))

        outfile.write('\n')




