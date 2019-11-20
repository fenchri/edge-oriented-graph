#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13/05/2019

author: fenia
"""

import argparse


def prf(tp, fp, fn):
    micro_p = float(tp) / (tp + fp) if (tp + fp != 0) else 0.0
    micro_r = float(tp) / (tp + fn) if (tp + fn != 0) else 0.0
    micro_f = ((2 * micro_p * micro_r) / (micro_p + micro_r)) if micro_p != 0.0 and micro_r != 0.0 else 0.0
    return [micro_p, micro_r, micro_f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str)
    parser.add_argument('--pred', type=str)
    parser.add_argument('--label', type=str)
    args = parser.parse_args()

    with open(args.pred) as pred, open(args.gold) as gold:
        preds_all = []
        preds_intra = []
        preds_inter = []

        golds_all = []
        golds_intra = []
        golds_inter = []

        for line in pred:
            line = line.rstrip().split('|')
            if line[5] == args.label:

                if (line[0], line[1], line[2], line[3], line[5]) not in preds_all:
                    preds_all += [(line[0], line[1], line[2], line[3], line[5])]

                if ((line[0], line[1], line[2], line[5]) not in preds_inter) and (line[3] == 'CROSS'):
                    preds_inter += [(line[0], line[1], line[2], line[5])]

                if ((line[0], line[1], line[2], line[5]) not in preds_intra) and (line[3] == 'NON-CROSS'):
                    preds_intra += [(line[0], line[1], line[2], line[5])]

        for line2 in gold:
            line2 = line2.rstrip().split('|')

            if line2[4] == args.label:

                if (line2[0], line2[1], line2[2], line2[3], line2[4]) not in golds_all:
                    golds_all += [(line2[0], line2[1], line2[2], line2[3], line2[4])]

                if ((line2[0], line2[1], line2[2], line2[4]) not in golds_inter) and (line2[3] == 'CROSS'):
                    golds_inter += [(line2[0], line2[1], line2[2], line2[4])]

                if ((line2[0], line2[1], line2[2], line2[4]) not in golds_intra) and (line2[3] == 'NON-CROSS'):
                    golds_intra += [(line2[0], line2[1], line2[2], line2[4])]

        tp = len([a for a in preds_all if a in golds_all])
        tp_intra = len([a for a in preds_intra if a in golds_intra])
        tp_inter = len([a for a in preds_inter if a in golds_inter])

        fp = len([a for a in preds_all if a not in golds_all])
        fp_intra = len([a for a in preds_intra if a not in golds_intra])
        fp_inter = len([a for a in preds_inter if a not in golds_inter])

        fn = len([a for a in golds_all if a not in preds_all])
        fn_intra = len([a for a in golds_intra if a not in preds_intra])
        fn_inter = len([a for a in golds_inter if a not in preds_inter])

        r1 = prf(tp, fp, fn)
        r2 = prf(tp_intra, fp_intra, fn_intra)
        r3 = prf(tp_inter, fp_inter, fn_inter)

        print('                                          TOTAL\tTP\tFP\tFN')
        print('Overall P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r1[0], r1[1], r1[2],
                                                                                               tp + fn, tp, fp, fn))
        print('INTRA P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r2[0], r2[1], r2[2],
                                                                                             tp_intra + fn_intra,
                                                                                             tp_intra, fp_intra,
                                                                                             fn_intra))
        print('INTER P/R/F1\t{:.4f}\t{:.4f}\t{:.4f}\t| {}\t{}\t{}\t{}'.format(r3[0], r3[1], r3[2],
                                                                                             tp_inter + fn_inter,
                                                                                             tp_inter, fp_inter,
                                                                                             fn_inter))


if __name__ == "__main__":
    main()
