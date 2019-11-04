#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/05/2019

author: fenia
"""

import os, re, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--systems', nargs='*')
parser.add_argument('--fin', type=str)
args = parser.parse_args()

pairs = {}
for s in args.systems:
    with open(s, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line in pairs:
                pairs[line] += 1
            else:
                pairs[line] = 1


with open(args.fin, 'w') as outfile:
    for p in pairs:
        if pairs[p] >= 3:
            outfile.write(p+'\n')
