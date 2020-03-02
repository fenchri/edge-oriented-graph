#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/05/2019

author: fenia
"""

import argparse
import numpy as np
from collections import OrderedDict
from recordtype import recordtype

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
args = parser.parse_args()


EntityInfo = recordtype('EntityInfo', 'type mstart mend sentNo')
PairInfo = recordtype('PairInfo', 'type direction cross closeA closeB')


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        yield l[i:i + n]


for d, data in zip(['DATA'], [args.data]):
    documents = {}
    entities = {}
    relations = {}

    with open(data, 'r') as infile:

        for line in infile:
            line = line.rstrip().split('\t')
            pairs = chunks(line[2:], 17)

            id_ = line[0]

            if id_ not in documents:
                documents[id_] = []

            for sent in line[1].split('|'):
                documents[id_] += [sent]

            if id_ not in entities:
                entities[id_] = OrderedDict()

            if id_ not in relations:
                relations[id_] = OrderedDict()

            for p in pairs:
                # pairs
                if (p[5], p[11]) not in relations[id_]:
                    relations[id_][(p[5], p[11])] = PairInfo(p[0], p[1], p[2], p[3], p[4])
                else:
                    print('duplicates!')

                # entities
                if p[5] not in entities[id_]:
                    entities[id_][p[5]] = EntityInfo(p[7], p[8], p[9], p[10])

                if p[11] not in entities[id_]:
                    entities[id_][p[11]] = EntityInfo(p[13], p[14], p[15], p[16])

    docs = len(documents)
    pair_types = {}
    inter_types = {}
    intra_types = {}
    ent_types = {}
    men_types = {}
    dist = {}
    for id_ in relations.keys():
        for k, p in relations[id_].items():

            if p.type not in pair_types:
                pair_types[p.type] = 0
            pair_types[p.type] += 1

            if p.type not in dist:
                dist[p.type] = []

            if p.type not in inter_types:
                inter_types[p.type] = 0

            if p.type not in intra_types:
                intra_types[p.type] = 0

            if p.type not in dist:
                dist[p.type] = []

            if p.cross == 'CROSS':
                inter_types[p.type] += 1
            else:
                intra_types[p.type] += 1

            if p.cross == 'CROSS':
                dist_ = 10000

                for m1 in entities[id_][k[0]].sentNo.split(':'):
                    for m2 in entities[id_][k[1]].sentNo.split(':'):

                        if abs(int(m1) - int(m2)) < dist_:
                            dist_ = abs(int(m1) - int(m2))

                dist[p.type] += [dist_]

        for e in entities[id_].values():
            if e.type not in ent_types:
                ent_types[e.type] = 0
            ent_types[e.type] += 1

            if e.type not in men_types:
                men_types[e.type] = 0
            for m in e.mstart.split(':'):
                men_types[e.type] += 1

    ents_per_doc = [len(entities[n]) for n in documents.keys()]
    ments_per_doc = [np.sum([len(e.sentNo.split(':')) for e in entities[n].values()]) for n in documents.keys()]
    ments_per_ent = [[len(e.sentNo.split(':')) for e in entities[n].values()] for n in documents.keys()]
    sents_per_doc = [len(s) for s in documents.values()]
    sent_len = [len(a.split()) for s in documents.values() for a in s]

    # write data
    with open('/'.join(args.data.split('/')[:-1]) + '/' + args.data.split('/')[-1].split('.')[0] + '.gold', 'w') as outfile:
        for id_ in relations.keys():
            for k, p in relations[id_].items():
                PairInfo = recordtype('PairInfo', 'type direction cross closeA closeB')
                outfile.write('{}|{}|{}|{}|{}\n'.format(id_, k[0], k[1], p.cross, p.type))

    print('''
    ----------------------- {} ----------------------
    Documents                       {}
    '''.format(d, docs))

    print('    Pairs')

    for x in ['{:<10}\t{:<5}'.format(k, v) for k, v in sorted(pair_types.items())]:
        print('                                    {}'.format(x))
    print()

    print('    Entities')
    for x in ['{:<10}\t{:<5}'.format(k, v) for k, v in sorted(ent_types.items())]:
        print('                                    {}'.format(x))
    print()

    print('    Mentions')
    for x in ['{:<10}\t{:<5}'.format(k, v) for k, v in sorted(men_types.items())]:
        print('                                    {}'.format(x))
    print()

    print('    Intra Pairs')
    for x in ['{:<10}\t{:<5}'.format(k, v) for k, v in sorted(intra_types.items())]:
        print('                                    {}'.format(x))
    print()

    print('    Inter Pairs')
    for x in ['{:<10}\t{:<5}'.format(k, v) for k, v in sorted(inter_types.items())]:
        print('                                    {}'.format(x))
    print()

    print('    Average/Max Sentence Distance')
    for x in ['{:<10}\t{:.1f}\t{}'.format(k, np.average(v), np.max(v)) for k, v in sorted(dist.items())]:
        print('                                    {}'.format(x))
    print()

    print('''
    Average entites/doc             {:.1f}
    Max                             {}
    
    Average mentions/doc            {:.1f}
    Max                             {}
    
    Average mentions/entity         {:.1f}
    Max                             {}
    
    Average sents/doc               {:.1f}
    Max                             {}
    
    Average/max sent length         {:.1f}
    Max                             {}
    '''.format(np.average(ents_per_doc),
               np.max(ents_per_doc),
               np.average(ments_per_doc),
               np.max(ments_per_doc),
               np.average([item for sublist in ments_per_ent for item in sublist]),
               np.max([item for sublist in ments_per_ent for item in sublist]),
               np.average(sents_per_doc),
               np.max(sents_per_doc),
               np.average(sent_len),
               np.max(sent_len)))













