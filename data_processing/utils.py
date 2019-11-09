#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: fenia
"""

import networkx


def to_graph(l):
    """
    https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    """
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also implies a number of edges:
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def using_split2(line, _len=len):
    """
    Credits to https://stackoverflow.com/users/1235039/aquavitae

    :param line: sentence
    :return: a list of words and their indexes in a string.
    """
    words = line.split(' ')
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset))
    return offsets


def replace2symbol(string):
    string = string.replace('”', '"').replace('’', "'").replace('–', '-').replace('‘', "'").replace('‑', '-').replace(
        '\x92', "'").replace('»', '"').replace('—', '-').replace('\uf8fe', ' ').replace('«', '"').replace(
        '\uf8ff', ' ').replace('£', '#').replace('\u2028', ' ').replace('\u2029', ' ')

    return string


def replace2space(string):
    spaces = ["\r", '\xa0', '\xe2\x80\x85', '\xc2\xa0', '\u2009', '\u2002', '\u200a', '\u2005', '\u2003', '\u2006',
              'Ⅲ', '…', 'Ⅴ', "\u202f"]

    for i in spaces:
        string = string.replace(i, ' ')
    return string

