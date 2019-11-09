# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 10:32:10 2017

@author: fenia
"""

from gensim.models.keyedvectors import KeyedVectors
import sys

"""
Transform from 'bin' to 'txt' word vectors.
Input: the bin file
Output: the txt file
"""
inp = sys.argv[1]
out = ''.join(inp.split('.bin')[:-1])+'.txt'

model = KeyedVectors.load_word2vec_format(inp, binary=True)
model.save_word2vec_format(out, binary=False)
