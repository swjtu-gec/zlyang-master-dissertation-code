#!/usr/bin/env python

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.segment import segment_sen

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', required=True, help='path to input file (output of fairseq)')
parser.add_argument('-o', '--output-file', dest="out_file", required=True, help="path to reformatted file")
parser.add_argument('--debpe',  action='store_true', help='whether to remove BPE segmentation.')
parser.add_argument('--char-seg', action='store_true', help='whether to re-seg sentence in char level')

args = parser.parse_args()


scount = -1
with open(args.input_file) as f, open(args.out_file, 'w') as out:
    for line in f:
        line = line.strip()
        pieces = line.split('\t')
        if pieces[0] == 'O':
            scount += 1
        if pieces[0] == 'H':
            hyp = pieces[2]
            if args.debpe:
                hyp = hyp.replace('@@ ','')
            if args.char_seg:
                hyp = hyp.replace(' ', '')
                hyp = ' '.join(segment_sen(hyp, True, False))
            score = pieces[1]
            to_write = "%d ||| %s ||| F0= %s ||| %s" % (scount, hyp, score, score)
            out.write(to_write + '\n')

