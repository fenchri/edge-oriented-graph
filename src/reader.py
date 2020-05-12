#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/06/2019

author: fenia
"""

from collections import OrderedDict
from recordtype import recordtype
import numpy as np


EntityInfo = recordtype('EntityInfo', 'id type mstart mend sentNo')
PairInfo = recordtype('PairInfo', 'type direction cross')


def chunks(l, n):
    """
    Successive n-sized chunks from l.
    """
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def overlap_chunk(chunk=1, lst=None):
    if len(lst) <= chunk:
        return [lst]
    else:
        return [lst[i:i + chunk] for i in range(0, len(lst)-chunk+1, 1)]


def read_subdocs(input_file, window, documents, entities, relations):
    """
    Read documents as sub-documents of N consecutive sentences.
    Args:
       input_file: file with documents
    """
    lost_pairs, total_pairs = 0, 0
    lengths = []
    sents = []
    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.rstrip().split('\t')
            pmid = line[0]
            text = line[1]
            prs = chunks(line[2:], 17)  # all the pairs in the document

            sentences = text.split('|')      # document sentences
            all_sent_lengths = [len(s.split(' ')) for s in sentences]  # document sentence lengths

            sent_chunks = overlap_chunk(chunk=window, lst=sentences)   # split document into sub-documents

            unique_pairs = []
            for num, sent in enumerate(sent_chunks):
                sent_ids = list(np.arange(int(window)) + num)

                sub_pmid = pmid+'__'+str(num)

                if sub_pmid not in documents:
                    documents[sub_pmid] = [t.split(' ') for t in sent]

                if sub_pmid not in entities:
                    entities[sub_pmid] = OrderedDict()

                if sub_pmid not in relations:
                    relations[sub_pmid] = OrderedDict()

                lengths += [max([len(d) for d in documents[sub_pmid]])]
                sents += [len(sent)]

                for p in prs:
                    # entities
                    for (ent, typ_, start, end, sn) in [(p[5], p[7], p[8], p[9], p[10]),
                                                        (p[11], p[13], p[14], p[15], p[16])]:

                        if ent not in entities[sub_pmid]:
                            s_ = list(map(int, sn.split(':')))          # doc-level ids
                            m_s_ = list(map(int, start.split(':')))
                            m_e_ = list(map(int, end.split(':')))
                            assert len(s_) == len(m_s_) == len(m_e_)

                            sent_no_new = []
                            mstart_new = []
                            mend_new = []
                            for n, (old_s, old_ms, old_me) in enumerate(zip(s_, m_s_, m_e_)):
                                if old_s in sent_ids:
                                    sub_ = sum(all_sent_lengths[0:old_s])

                                    assert sent[old_s-num] == sentences[old_s]
                                    assert sent[old_s-num].split(' ')[(old_ms-sub_):(old_me-sub_)] == \
                                        ' '.join(sentences).split(' ')[old_ms:old_me]
                                    sent_no_new += [old_s - num]
                                    mstart_new += [old_ms - sub_]
                                    mend_new += [old_me - sub_]

                            if sent_no_new and mstart_new and mend_new:
                                entities[sub_pmid][ent] = EntityInfo(ent, typ_,
                                                                     ':'.join(map(str, mstart_new)),
                                                                     ':'.join(map(str, mend_new)),
                                                                     ':'.join(map(str, sent_no_new)))

                for p in prs:
                    # pairs
                    if (p[5] in entities[sub_pmid]) and (p[11] in entities[sub_pmid]):
                        if (p[5], p[11]) not in relations[sub_pmid]:
                            relations[sub_pmid][(p[5], p[11])] = PairInfo(p[0], p[1], p[2])

                            if (pmid, p[5], p[11]) not in unique_pairs:
                                unique_pairs += [(pmid, p[5], p[11])]

            if len(prs) != len(unique_pairs):
                for x in prs:
                    if (pmid, x[5], x[11]) not in unique_pairs:
                        if x[0] != '1:NR:2' and x[0] != 'not_include':
                            lost_pairs += 1
                            print('--> Lost pair {}, {}, {}: {} {}'.format(pmid, x[5], x[11], x[10], x[16]))
                    else:
                        if x[0] != '1:NR:2' and x[0] != 'not_include':
                            total_pairs += 1

    todel = []
    for pmid, d in relations.items():
        if not relations[pmid]:
            todel += [pmid]

    for pmid in todel:
        del documents[pmid]
        del entities[pmid]
        del relations[pmid]

    print('LOST PAIRS: {}/{}'.format(lost_pairs, total_pairs))
    assert len(entities) == len(documents) == len(relations)
    return lengths, sents, documents, entities, relations


def read(input_file, documents, entities, relations):
    """
    Read the full document at a time.
    """
    lengths = []
    sents = []
    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.rstrip().split('\t')
            pmid = line[0]
            text = line[1]
            prs = chunks(line[2:], 17)

            if pmid not in documents:
                documents[pmid] = [t.split(' ') for t in text.split('|')]

            if pmid not in entities:
                entities[pmid] = OrderedDict()

            if pmid not in relations:
                relations[pmid] = OrderedDict()

            # max intra-sentence length and max inter-sentence length
            lengths += [max([len(s) for s in documents[pmid]] + [len(documents[pmid])])]
            sents += [len(text.split('|'))]

            allp = 0
            for p in prs:
                if (p[5], p[11]) not in relations[pmid]:
                    relations[pmid][(p[5], p[11])] = PairInfo(p[0], p[1], p[2])
                    allp += 1
                else:
                    print('duplicates!')

                # entities
                if p[5] not in entities[pmid]:
                    entities[pmid][p[5]] = EntityInfo(p[5], p[7], p[8], p[9], p[10])

                if p[11] not in entities[pmid]:
                    entities[pmid][p[11]] = EntityInfo(p[11], p[13], p[14], p[15], p[16])

            assert len(relations[pmid]) == allp

    todel = []
    for pmid, d in relations.items():
        if not relations[pmid]:
            todel += [pmid]

    for pmid in todel:
        del documents[pmid]
        del entities[pmid]
        del relations[pmid]

    return lengths, sents, documents, entities, relations
