#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/09/2019

author: fenia
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_train', type=str)
parser.add_argument('--output_dev', type=str)
parser.add_argument('--list', type=str)
args = parser.parse_args()


with open(args.list, 'r') as infile:
    docs = [i.rstrip() for i in infile]
docs = frozenset(docs)

with open(args.input_file, 'r') as infile, open(args.output_train, 'w') as otr, open(args.output_dev, 'w') as odev:
    for line in infile:
        pmid = line.rstrip().split('\t')[0]

        if pmid in docs:
            otr.write(line)
        else:
            odev.write(line)

