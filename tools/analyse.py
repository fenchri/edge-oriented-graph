#!/usr/bin/python3

import os
import sys
import argparse
from collections import OrderedDict
from recordtype import recordtype
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gold', type=str)
parser.add_argument('--pred', type=str)
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


documents = {}
entities = {}
relations = {}
with open(args.gold, 'r') as gold_file:
    for line in gold_file:
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
all_pairs = {}
for id_ in relations.keys():
    for k, p in relations[id_].items():
        pair_ = (id_, k[0], k[1], p.cross, p.type)
        all_pairs[pair_] = {}

        # append the distance between their mentions
        if p.cross == 'CROSS':
            dist_ = 1000000
            for m1 in entities[id_][k[0]].sentNo.split(':'):
                for m2 in entities[id_][k[1]].sentNo.split(':'):
                    if abs(int(m1) - int(m2)) < dist_:
                        dist_ = abs(int(m1) - int(m2))
            all_pairs[pair_]['dist'] = dist_
        else:
            all_pairs[pair_]['dist'] = 0

        all_pairs[pair_]['ents'] = len(entities[id_])
        
        all_pairs[pair_]['ments'] = np.sum([len(e.sentNo.split(':')) for e in entities[id_].values()]) 
        all_pairs[pair_]['sents'] = len(documents[id_])
        

# distance between inter-sentence pairs (in terms of sentences)
for d in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
    with open('temporary_gold.txt', 'w') as temp_gold, open(args.pred, 'r') as pred_file, open('temporary_pred.txt', 'w') as temp_pred:
        for ap in all_pairs.keys():
            if all_pairs[ap]['ments'] < d:
                if ap[4] != '1:NR:2' and ap[4] != 'not_include':
                    temp_gold.write('{}|{}|{}|{}|{}\n'.format(ap[0], ap[1], ap[2], ap[3], ap[4]))


        for line in pred_file:
            line = line.rstrip().split('|')
            pmid = line[0]
            arg1 = line[1]
            arg2 = line[2]
            cr = line[3]
            typ_ = line[4]
            
            for ap in all_pairs.keys():
                if (pmid == ap[0]) and (arg1 == ap[1]) and (arg2 == ap[2]) and (cr == ap[3]):
                    temp_pred.write('{}\n'.format('|'.join(line)))

    print('Performance for INTER DISTANCE = {} sentences'.format(d))
    os.system('python3 eval.py temporary_pred.txt temporary_gold.txt')
            
