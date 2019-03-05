#!/usr/bin/python
# encoding: utf-8

from __future__ import unicode_literals, print_function
from pkunlp import Segmentor, NERTagger, POSTagger

import argparse
import time
import codecs, json, re, sys, time
import multiprocessing

# usage:
# python pkunlp_segmenter.py --corpus data.train.src  --segsuffix seg

def parseargs():
    parser = argparse.ArgumentParser(description="segment corpus")

    parser.add_argument("--corpus", required=True,
                        help="input corpora")
    parser.add_argument("--segsuffix", type=str, default="seg",
                        help="Suffix of output files")
    return parser.parse_args()


if __name__ == "__main__":
    print("Start processing")
    start_time = time.time()
    parsed_args = parseargs()

    segmentor = Segmentor("feature/segment.feat", "feature/segment.dic")

    with open(parsed_args.corpus ,'r',encoding='utf-8') as corpus_f,\
            open(parsed_args.corpus + "." + parsed_args.segsuffix,'w',encoding='utf-8',errors='ignore') as seg_output_f:
        for line in corpus_f:
            if len(line) <= 1500 and len(line) != 0:
                segments = segmentor.seg_string(line.strip())
                segments_str = " ".join(segments)
                seg_output_f.write(segments_str + "\n")
    print("Done in", time.time()-start_time, "seconds")