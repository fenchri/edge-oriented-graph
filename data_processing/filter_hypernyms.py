#!/usr/bin/env python3
"""
Created on 17/04/19

author: fenia
"""

import argparse
import codecs
from collections import defaultdict
'''
Adaptation of https://github.com/patverga/bran/blob/master/src/processing/utils/filter_hypernyms.py
'''

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', required=True, help='input file in 13col tsv')
parser.add_argument('-m', '--mesh_file', required=True, help='mesh file to get hierarchy from')
parser.add_argument('-o', '--output_file', required=True, help='write results to this file')

args = parser.parse_args()


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        yield l[i:i + n]


# read in mesh hierarchy
ent_tree_map = defaultdict(list)
with codecs.open(args.mesh_file, 'r') as f:
    lines = [l.rstrip().split('\t') for i, l in enumerate(f) if i > 0]
    [ent_tree_map[l[1]].append(l[0]) for l in lines]


# read in positive input file and organize by document
print('Loading examples from %s' % args.input_file)
pos_doc_examples = defaultdict(list)
neg_doc_examples = defaultdict(list)

unfilitered_pos_count = 0
unfilitered_neg_count = 0
text = {}
with open(args.input_file, 'r') as f:
    lines = [l.strip().split('\t') for l in f]

    for l in lines:
        pmid = l[0]
        text[pmid] = pmid+'\t'+l[1]

        for r in chunks(l[2:], 17):

            if r[0] == '1:NR:2':
                assert ((r[7] == 'Chemical') and (r[13] == 'Disease'))
                neg_doc_examples[pmid].append(r)
                unfilitered_neg_count += 1
            elif r[0] == '1:CID:2':
                assert ((r[7] == 'Chemical') and (r[13] == 'Disease'))
                pos_doc_examples[pmid].append(r)
                unfilitered_pos_count += 1


# iterate over docs
hypo_count = 0
negative_count = 0

all_pos = 0
with open(args.output_file, 'w') as out_f:
    for doc_id in pos_doc_examples.keys():
        towrite = text[doc_id]

        for r in pos_doc_examples[doc_id]:
            towrite += '\t'
            towrite += '\t'.join(r)
        all_pos += len(pos_doc_examples[doc_id])

        # get nodes for all the positive diseases
        pos_e2_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id]
                           for pos_node in ent_tree_map[pe[11]]]

        pos_e1_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id]
                           for pos_node in ent_tree_map[pe[5]]]

        filtered_neg_exampled = []
        for ne in neg_doc_examples[doc_id]:
            neg_e1 = ne[5]
            neg_e2 = ne[11]
            example_hyponyms = 0
            for neg_node in ent_tree_map[ne[11]]:
                hyponyms = [pos_node for pos_node, pe in pos_e2_examples
                            if neg_node in pos_node and neg_e1 == pe[5]] \
                           + [pos_node for pos_node, pe in pos_e1_examples
                              if neg_node in pos_node and neg_e2 == pe[11]]
                example_hyponyms += len(hyponyms)
            if example_hyponyms == 0:
                towrite += '\t'+'\t'.join(ne)
                negative_count += 1
            else:
                ne[0] = 'not_include'   # just don't include the negative pairs, but keep the entities
                towrite += '\t'+'\t'.join(ne)
                hypo_count += example_hyponyms
        out_f.write(towrite+'\n')

print('Mesh entities: %d' % len(ent_tree_map))
print('Positive Docs: %d' % len(pos_doc_examples))
print('Negative Docs: %d' % len(neg_doc_examples))
print('Positive Count: %d   Initial Negative Count: %d   Final Negative Count: %d   Hyponyms: %d' %
(unfilitered_pos_count, unfilitered_neg_count, negative_count, hypo_count))
print(all_pos)