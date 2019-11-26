#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: fenia
"""

import os
import re
from tqdm import tqdm
from recordtype import recordtype
from collections import OrderedDict
import argparse
import pickle
from itertools import permutations, combinations
from tools import sentence_split_genia, tokenize_genia
from tools import adjust_offsets, find_mentions, find_cross, fix_sent_break, convert2sent, generate_pairs
from readers import *

TextStruct = recordtype('TextStruct', 'pmid txt')
EntStruct = recordtype('EntStruct', 'pmid name off1 off2 type kb_id sent_no word_id bio')
RelStruct = recordtype('RelStruct', 'pmid type arg1 arg2')


def main():
    """ 
    Main processing function 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--output_file', '-o', type=str)
    parser.add_argument('--data', '-d', type=str)
    args = parser.parse_args()

    if args.data == 'GDA':
        abstracts, entities, relations = readPubTator(args)
        type1 = ['Gene']
        type2 = ['Disease']

    elif args.data == 'CDR':
        abstracts, entities, relations = readPubTator(args)
        type1 = ['Chemical']
        type2 = ['Disease']

    else:
        print('Dataset non-existent.')
        sys.exit()

    if not os.path.exists(args.output_file + '_files/'):
        os.makedirs(args.output_file + '_files/')

    # Process
    positive, negative = 0, 0
    with open(args.output_file + '.data', 'w') as data_out:
        pbar = tqdm(list(abstracts.keys()))
        for i in pbar:
            pbar.set_description("Processing Doc_ID {}".format(i))

            ''' Sentence Split '''
            orig_sentences = [item for sublist in [a.txt.split('\n') for a in abstracts[i]] for item in sublist]
            split_sents = sentence_split_genia(orig_sentences)
            split_sents = fix_sent_break(split_sents, entities[i])
            with open(args.output_file + '_files/' + i + '.split.txt', 'w') as f:
                f.write('\n'.join(split_sents))

            # adjust offsets
            new_entities = adjust_offsets(orig_sentences, split_sents, entities[i], show=False)

            ''' Tokenisation '''
            token_sents = tokenize_genia(split_sents)
            with open(args.output_file + '_files/' + i + '.split.tok.txt', 'w') as f:
                f.write('\n'.join(token_sents))

            # adjust offsets
            new_entities = adjust_offsets(split_sents, token_sents, new_entities, show=True)

            ''' Find mentions '''
            unique_entities = find_mentions(new_entities)
            with open(args.output_file + '_files/' + i + '.mention', 'wb') as f:
                pickle.dump(unique_entities, f, pickle.HIGHEST_PROTOCOL)

            ''' Generate Pairs '''
            if i in relations:
                pairs = generate_pairs(unique_entities, type1, type2, relations[i])
            else:
                pairs = generate_pairs(unique_entities, type1, type2, [])   # generate only negative pairs

            # 'pmid type arg1 arg2 dir cross'
            data_out.write('{}\t{}'.format(i, '|'.join(token_sents)))

            for args_, p in pairs.items():
                if p.type != '1:NR:2':
                    positive += 1
                elif p.type == '1:NR:2':
                    negative += 1

                data_out.write('\t{}\t{}\t{}\t{}-{}\t{}-{}'.format(p.type, p.dir, p.cross, p.closest[0].word_id[0],
                                                                                           p.closest[0].word_id[-1]+1,
                                                                                           p.closest[1].word_id[0],
                                                                                           p.closest[1].word_id[-1]+1))
                data_out.write('\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    '|'.join([g for g in p.arg1]),
                    '|'.join([e.name for e in unique_entities[p.arg1]]),
                    unique_entities[p.arg1][0].type,
                    ':'.join([str(e.word_id[0]) for e in unique_entities[p.arg1]]),
                    ':'.join([str(e.word_id[-1] + 1) for e in unique_entities[p.arg1]]),
                    ':'.join([str(e.sent_no) for e in unique_entities[p.arg1]])))

                data_out.write('\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    '|'.join([g for g in p.arg2]),
                    '|'.join([e.name for e in unique_entities[p.arg2]]),
                    unique_entities[p.arg2][0].type,
                    ':'.join([str(e.word_id[0]) for e in unique_entities[p.arg2]]),
                    ':'.join([str(e.word_id[-1] + 1) for e in unique_entities[p.arg2]]),
                    ':'.join([str(e.sent_no) for e in unique_entities[p.arg2]])))
            data_out.write('\n')
    print('Total positive pairs:', positive)
    print('Total negative pairs:', negative)


if __name__ == "__main__":
    main()
