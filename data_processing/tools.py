#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: fenia
"""

import os
import sys
import re
from recordtype import recordtype
from networkx.algorithms.components.connected import connected_components
from itertools import combinations
import numpy as np
from collections import OrderedDict
from utils import to_graph, to_edges, using_split2
from tqdm import tqdm
sys.path.append('./common/genia-tagger-py/')
from geniatagger import GENIATagger

pwd = '/'.join(os.path.realpath(__file__).split('/')[:-1])

genia_splitter = os.path.join("./common", "geniass")
genia_tagger = GENIATagger(os.path.join("./common", "genia-tagger-py", "geniatagger-3.0.2", "geniatagger"))


TextStruct = recordtype('TextStruct', 'pmid txt')
EntStruct = recordtype('EntStruct', 'pmid name off1 off2 type kb_id sent_no word_id bio')
RelStruct = recordtype('RelStruct', 'pmid type arg1 arg2')
PairStruct = recordtype('PairStruct', 'pmid type arg1 arg2 dir cross closest')


def generate_pairs(uents, type1, type2, true_rels):
    """
    Generate pairs (both positive & negative):
    Type1 - Type2 should have 1-1 association, e.g. [A, A] [B, C] --> (A,B), (A,C)
    Args:
        uents:
        type1: (list) with entity semantic types
        type2: (list) with entity semantic types
        true_rels:
    """
    pairs = OrderedDict()
    combs = combinations(uents, 2)

    unk = 0
    total_rels = len(true_rels)
    found_rels = 0

    for c in combs:
        # all pairs
        diff = 99999

        target = []
        for e1 in uents[c[0]]:
            for e2 in uents[c[1]]:
                # find most close pair to each other
                if e1.word_id[-1] <= e2.word_id[0]:
                    if abs(e2.word_id[0] - e1.word_id[-1]) < diff:
                        target = [e1, e2]
                        diff = abs(e2.word_id[0] - e1.word_id[-1])
                else:
                    if abs(e1.word_id[0] - e2.word_id[-1]) < diff:
                        target = [e1, e2]
                        diff = abs(e2.word_id[0] - e1.word_id[-1])

        if target[0].word_id[-1] <= target[1].word_id[0]:  # A before B (in text)
            a1 = target[0]
            a2 = target[1]
        else:                                              # B before A (in text)
            a1 = target[1]
            a2 = target[0]

        if c[0][0].startswith('UNK:') or c[1][0].startswith('UNK:'):  # ignore non-grounded entities
            continue

        cross_res = find_cross(c, uents)
        not_found_rels = 0

        for tr in true_rels:

            # AB existing relation
            if list(set(tr.arg1).intersection(set(c[0]))) and list(set(tr.arg2).intersection(set(c[1]))):
                for t1, t2 in zip(type1, type2):
                    if uents[c[0]][0].type == t1 and uents[c[1]][0].type == t2:
                        pairs[(c[0], c[1])] = \
                            PairStruct(tr.pmid, '1:' + tr.type + ':2', c[0], c[1], 'L2R', cross_res, (a1, a2))
                        found_rels += 1

            # BA existing relation
            elif list(set(tr.arg1).intersection(set(c[1]))) and list(set(tr.arg2).intersection(set(c[0]))):
                for t1, t2 in zip(type1, type2):
                    if uents[c[1]][0].type == t1 and uents[c[0]][0].type == t2:
                        pairs[(c[1], c[0])] = \
                            PairStruct(tr.pmid, '1:'+tr.type+':2', c[1], c[0], 'R2L', cross_res, (a2, a1))
                        found_rels += 1

            # relation not found
            else:
                not_found_rels += 1

        # this pair does not have a relation
        if not_found_rels == total_rels:
            for t1, t2 in zip(type1, type2):
                if uents[c[0]][0].type == t1 and uents[c[1]][0].type == t2:
                    pairs[(c[0], c[1])] = PairStruct(a1.pmid, '1:NR:2', c[0], c[1], 'L2R', cross_res, (a1, a2))
                    unk += 1
                elif uents[c[1]][0].type == t1 and uents[c[0]][0].type == t2:
                    pairs[(c[1], c[0])] = PairStruct(a1.pmid, '1:NR:2', c[1], c[0], 'R2L', cross_res, (a2, a1))
                    unk += 1

    assert found_rels == total_rels, '{} <> {}, {}, {}'.format(found_rels, total_rels, true_rels, pairs)

    # # Checking
    # if found_rels != total_rels:
    #     print('NON-FOUND RELATIONS: {} <> {}'.format(found_rels, total_rels))
    #     for p in true_rels:
    #         if (p.arg1, p.arg2) not in pairs:
    #             print(p.arg1, p.arg2)
    return pairs


def convert2sent(arg1, arg2, token_sents):
    """
    Convert document info to sentence (for pairs in same sentence).
    Args:
        arg1:
        arg2:
        token_sents:
    """
    # make sure they are in the same sentence
    assert arg1.sent_no == arg2.sent_no, 'error: entities not in the same sentence'

    toks_per_sent = []
    sent_offs = []
    cnt = 0
    for i, s in enumerate(token_sents):
        toks_per_sent.append(len(s.split(' ')))
        sent_offs.append((cnt, cnt+len(s.split(' '))-1))
        cnt = len(' '.join(token_sents[:i+1]).split(' '))

    target_sent = token_sents[arg1.sent_no].split(' ')
    n = sum(toks_per_sent[0:arg1.sent_no])

    arg1_span = [a-n for a in arg1.word_id]
    arg2_span = [a-n for a in arg2.word_id]
    assert target_sent[arg1_span[0]:arg1_span[-1]+1] == \
        ' '.join(token_sents).split(' ')[arg1.word_id[0]:arg1.word_id[-1]+1]
    assert target_sent[arg2_span[0]:arg2_span[-1]+1] == \
        ' '.join(token_sents).split(' ')[arg2.word_id[0]:arg2.word_id[-1]+1]

    arg1_n = EntStruct(arg1.pmid, arg1.name, arg1.off1, arg1.off2, arg1.type,
                       arg1.kb_id, arg1.sent_no, arg1_span, arg1.bio)
    arg2_n = EntStruct(arg2.pmid, arg2.name, arg2.off1, arg2.off2, arg2.type,
                       arg2.kb_id, arg2.sent_no, arg2_span, arg2.bio)

    return arg1_n, arg2_n


def find_cross(pair, unique_ents):
    """
    Find if the pair is in cross or non-cross sentence.
    Args:
        pair: (tuple) target pair
        unique_ents: (dic) entities based on grounded IDs
    Returns: (str) cross/non-cross
    """
    non_cross = False
    for m1 in unique_ents[pair[0]]:
        for m2 in unique_ents[pair[1]]:
            if m1.sent_no == m2.sent_no:
                non_cross = True
    if non_cross:
        return 'NON-CROSS'
    else:
        return 'CROSS'


def fix_sent_break(sents, entities):
    """
    Fix sentence break + Find sentence of each entity
    Args:
        sents: (list) sentences
        entities: (recordtype)
    Returns: (list) sentences with fixed sentence breaks
    """
    sents_break = '\n'.join(sents)

    for e in entities:
        if '\n' in sents_break[e.off1:e.off2]:
            sents_break = sents_break[0:e.off1] + sents_break[e.off1:e.off2].replace('\n', ' ') + sents_break[e.off2:]
    return sents_break.split('\n')


def find_mentions(entities):
    """
    Find unique entities and their mentions
    Args:
        entities: (dic) a struct for each entity
    Returns: (dic) unique entities based on their grounded ID, if -1 ID=UNK:No
    """
    equivalents = []
    for e in entities:
        if e.kb_id not in equivalents:
            equivalents.append(e.kb_id)

    # mention-level data sets
    g = to_graph(equivalents)
    cc = connected_components(g)

    unique_entities = OrderedDict()
    unk_id = 0
    for c in cc:
        if tuple(c)[0] == '-1':
            continue
        unique_entities[tuple(c)] = []

    # consider non-grounded entities as separate entities
    for e in entities:
        if e.kb_id[0] == '-1':
            unique_entities[tuple(('UNK:' + str(unk_id),))] = [e]
            unk_id += 1
        else:
            for ue in unique_entities.keys():
                if list(set(e.kb_id).intersection(set(ue))):
                    unique_entities[ue] += [e]

    return unique_entities


def sentence_split_genia(tabst):
    """
    Sentence Splitting Using GENIA sentence splitter
    Args:
        tabst: (list) title+abstract

    Returns: (list) all sentences in abstract
    """
    os.chdir(genia_splitter)

    with open('temp_file.txt', 'w') as ofile:
        for t in tabst:
            ofile.write(t+'\n')
    os.system('./geniass temp_file.txt temp_file.split.txt > /dev/null 2>&1')

    split_lines = []
    with open('temp_file.split.txt', 'r') as ifile:
        for line in ifile:
            line = line.rstrip()
            if line != '':
                split_lines.append(line.rstrip())
    os.system('rm temp_file.txt temp_file.split.txt')
    os.chdir(pwd)
    return split_lines


def tokenize_genia(sents):
    """
    Tokenization using Genia Tokenizer
    Args:
        sents: (list) sentences

    Returns: (list) tokenized sentences
    """
    token_sents = []
    for i, s in enumerate(sents):
        tokens = []

        for word, base_form, pos_tag, chunk, named_entity in genia_tagger.tag(s):
            tokens += [word]

        text = []
        for t in tokens:
            if t == "'s":
                text.append(t)
            elif t == "''":
                text.append(t)
            else:
                text.append(t.replace("'", " ' "))

        text = ' '.join(text)
        text = text.replace("-LRB-", '(')
        text = text.replace("-RRB-", ')')
        text = text.replace("-LSB-", '[')
        text = text.replace("-RSB-", ']')
        text = text.replace("``", '"')
        text = text.replace("`", "'")
        text = text.replace("'s", " 's")
        text = text.replace('-', ' - ')
        text = text.replace('/', ' / ')
        text = text.replace('+', ' + ')
        text = text.replace('.', ' . ')
        text = text.replace('=', ' = ')
        text = text.replace('*', ' * ')
        if '&amp;' in s:
            text = text.replace("&", "&amp;")
        else:
            text = text.replace("&amp;", "&")

        text = re.sub(' +', ' ', text).strip()  # remove continuous spaces

        if "''" in ''.join(s):
            pass
        else:
            text = text.replace("''", '"')

        token_sents.append(text)
    return token_sents


def adjust_offsets(old_sents, new_sents, old_entities, show=False):
    """
    Adjust offsets based on tokenization
    Args:
        old_sents: (list) old, non-tokenized sentences
        new_sents: (list) new, tokenized sentences
        old_entities: (dic) entities with old offsets
    Returns:
        new_entities: (dic) entities with adjusted offsets
        abst_seq: (list) abstract sequence with entity tags
    """
    cur = 0
    new_sent_range = []
    for s in new_sents:
        new_sent_range += [(cur, cur + len(s))]
        cur += len(s) + 1

    original = " ".join(old_sents)
    newtext = " ".join(new_sents)
    new_entities = []
    terms = {}
    for e in old_entities:
        start = int(e.off1)
        end = int(e.off2)

        if (start, end) not in terms:
            terms[(start, end)] = [[start, end, e.type, e.name, e.pmid, e.kb_id]]
        else:
            terms[(start, end)].append([start, end, e.type, e.name, e.pmid, e.kb_id])

    orgidx = 0
    newidx = 0
    orglen = len(original)
    newlen = len(newtext)

    terms2 = terms.copy()
    while orgidx < orglen and newidx < newlen:
        # print(repr(original[orgidx]), orgidx, repr(newtext[newidx]), newidx)
        if original[orgidx] == newtext[newidx]:
            orgidx += 1
            newidx += 1
        elif original[orgidx] == "`" and newtext[newidx] == "'":
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == '\t':
            newidx += 1
        elif original[orgidx] == '\t':
            orgidx += 1
        elif newtext[newidx] == '.':
            # ignore extra "." for stanford
            newidx += 1
        else:
            print("Non-existent text: %d\t --> %s != %s " % (orgidx, repr(original[orgidx-10:orgidx+10]),
                                                             repr(newtext[newidx-10:newidx+10])))
            exit(0)

        starts = [key[0] for key in terms2.keys()]
        ends = [key[1] for key in terms2.keys()]

        if orgidx in starts:
            tt = [key for key in terms2.keys() if key[0] == orgidx]
            for sel in tt:
                for l in terms[sel]:
                    l[0] = newidx

        if orgidx in ends:
            tt2 = [key for key in terms2.keys() if key[1] == orgidx]
            for sel2 in tt2:
                for l in terms[sel2]:
                    if l[1] == orgidx:
                        l[1] = newidx

            for t_ in tt2:
                del terms2[t_]

    ent_sequences = []
    for ts in terms.values():
        for term in ts:
            condition = False

            if newtext[term[0]:term[1]].replace(" ", "").replace("\n", "") != term[3].replace(" ", "").replace('\n', ''):
                if newtext[term[0]:term[1]].replace(" ", "").replace("\n", "").lower() == \
                        term[3].replace(" ", "").replace('\n', '').lower():
                    condition = True
                    tqdm.write('DOC_ID {}, Lowercase Issue: {} <-> {}'.format(term[4], newtext[term[0]:term[1]], term[3]))
                else:
                    condition = False
            else:
                condition = True

            if condition:
                """ Convert to word Ids """
                tok_seq = []
                span2append = []
                bio = []
                tag = term[2]
               
                for tok_id, (tok, start, end) in enumerate(using_split2(newtext)):
                    start = int(start)
                    end = int(end)

                    if (start, end) == (term[0], term[1]):
                        bio.append('B-' + tag)
                        tok_seq.append('B-' + tag)
                        span2append.append(tok_id)

                    elif start == term[0] and end < term[1]:
                        bio.append('B-' + tag)
                        tok_seq.append('B-' + tag)
                        span2append.append(tok_id)

                    elif start > term[0] and end < term[1]:
                        bio.append('I-' + tag)
                        tok_seq.append('I-' + tag)
                        span2append.append(tok_id)

                    elif start > term[0] and end == term[1] and (start != end):
                        bio.append('I-' + tag)
                        tok_seq.append('I-' + tag)
                        span2append.append(tok_id)

                    elif len(set(range(start, end)).intersection(set(range(term[0], term[1])))) > 0:
                        span2append.append(tok_id)

                        if show:
                            tqdm.write('DOC_ID {}, entity: {:<20} ({:4}-{:4}) '
                                       '<-> token: {:<20} ({:4}-{:4}) <-> final: {:<20}'.format(
                                term[4], newtext[term[0]:term[1]], term[0], term[1], tok, start, end,
                                ' '.join(newtext.split(' ')[span2append[0]:span2append[-1]+1])))

                        if not bio:
                            bio.append('B-' + tag)
                            tok_seq.append('B-' + tag)
                        else:
                            bio.append('I-' + tag)
                            tok_seq.append('I-' + tag)

                    else:
                        tok_seq.append('O')

                ent_sequences += [tok_seq]

                # inlude all tokens!!
                if len(span2append) != len(newtext[term[0]:term[1]].split(' ')):
                    tqdm.write('DOC_ID {}, entity {}, tokens {}\n{}'.format(
                        term[4], newtext[term[0]:term[1]].split(' '), span2append, newtext))

                # Find sentence number of each entity
                sent_no = []
                for s_no, sr in enumerate(new_sent_range):
                    if set(np.arange(term[0], term[1])).issubset(set(np.arange(sr[0], sr[1]))):
                        sent_no += [s_no]

                assert (len(sent_no) == 1), '{} ({}, {}) -- {} -- {} <> {}'.format(sent_no, term[0], term[1],
                                                                                   new_sent_range,
                                                                                   newtext[term[0]:term[1]], term[3])

                new_entities += [EntStruct(term[4], newtext[term[0]:term[1]], term[0], term[1], term[2], term[5],
                                           sent_no[0], span2append, bio)]
            else:
                print(newtext, term[3])
                assert False, 'ERROR: {} ({}-{}) <=> {}'.format(repr(newtext[term[0]:term[1]]), term[0], term[1],
                                                                repr(term[3]))
    return new_entities
