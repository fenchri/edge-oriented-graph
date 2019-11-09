#!/usr/bin/python3

import numpy as np
import glob
import sys
from collections import OrderedDict
import argparse

"""
Crop embeddings to the size of the dataset, i.e. keeping only existing words.
"""

def load_pretrained_embeddings(embeds):
    """
        :param params: input parameters
        :returns
            dictionary with words (keys) and embeddings (values)
    """
    if embeds:
        E = OrderedDict()
        with open(embeds, 'r') as vectors:
            for x, line in enumerate(vectors):
                if x == 0 and len(line.split()) == 2:
                    words, num = map(int, line.rstrip().split())
                else:
                    word = line.rstrip().split()[0]
                    vec = line.rstrip().split()[1:]
                    n = len(vec)
                    if len(vec) != num:
                        # print('Wrong dimensionality: {} {} != {}'.format(word, len(vec), num))
                        continue
                    else:
                        E[word] = np.asarray(vec, dtype=np.float32)
        print('Pre-trained word embeddings: {} x {}'.format(len(E), n))
    else:
        E = OrderedDict()
        print('No pre-trained word embeddings loaded.')
    return E


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_embeds', type=str)
    parser.add_argument('--out_embeds', type=str)
    parser.add_argument('--in_data', nargs='+')
    args = parser.parse_args()

    words = []
    print('Extracting words from the dataset ... ', end="")

    for filef in args.in_data:
        with open(filef, 'r') as infile:
            for line in infile:
                line = line.strip().split('\t')[1]
                line = line.split('|')
                line = [l.split(' ') for l in line]
                line = [item for sublist in line for item in sublist]

                for l in line:
                    words.append(l)
    print('Done')

    # make lowercase
    words_lower = list(map(lambda x:x.lower(), words))

    print('Loading embeddings ... ', end="")
    embeddings = load_pretrained_embeddings(args.full_embeds)

    print('Writing final embeddings ... ', end="")
    words = set(words)
    words_lower = set(words_lower)  # lowercased

    new_embeds = OrderedDict()
    for w in embeddings.keys():
        if (w in words) or (w in words_lower):
            new_embeds[w] = embeddings[w]

    with open(args.out_embeds, 'w') as outfile:
        for g in new_embeds.keys():
            outfile.write('{} {}\n'.format(g, ' '.join(map(str, list(new_embeds[g])))))
    print('Done')
    
if __name__ == "__main__":
    main()
