"""
This script used to statistic corpus information including distribution of tokens and sentences.
"""
import argparse
import math
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils import reader
from utils.tools import my_getattr
from utils import tools


def main(args):
    if args.stat_tokens:
        stat_tokens_distribution(args.corpus_fname, args.encoding)
    elif args.stat_sens:
        stat_sens_distribution(args.corpus_fname, args.short, args.long, args.encoding)


def stat_tokens_distribution(corpus_fname, encoding='utf-8'):
    print('=================== tokens distribution ========================')
    with open(corpus_fname, 'r', encoding=encoding) as file:
        tcounts = Counter(file.read().split())
    tcounts = list(tcounts.items())
    times_stat = {}
    for token, count in tcounts:
        if times_stat.get(count) is None:
            times_stat[count] = 1
        else:
            times_stat[count] += 1
    vocab_size = len(tcounts)
    print('vocab size:', vocab_size)
    all_distinct_tokens = reader.read_tokens(corpus_fname, encoding)
    assert vocab_size == len(all_distinct_tokens), 'len of tcounts != corpus distinct tokens count'
    times_stat = list(times_stat.items())
    assert vocab_size == sum(map(lambda x: x[1], times_stat))
    all_tokens_num = sum(map(lambda x: x[1], tcounts))
    assert all_tokens_num == sum([times*cnt for times, cnt in times_stat])
    print('all tokens num:', all_tokens_num)
    times_stat.sort(key=lambda x: x[1], reverse=True)
    for times, cnt in times_stat:
        print('{0:6} tokens occur {1:10} times, the ratio of vocab_size is {2:>10.6f}% '
              '| the ratio of all tokens num is {3:>10.6f}%'
              .format(cnt, times, cnt / vocab_size * 100, cnt*times / all_tokens_num * 100))


def stat_sens_distribution(corpus_fname, short, long, encoding='utf-8'):
    lines_cnt = reader.count_lines(corpus_fname, encoding)
    print('=================== sentences distribution ========================')
    len_stat = {}
    with open(corpus_fname, 'r', encoding=encoding) as file:
        for line in file:
            sen_len = len(line.split())
            if sen_len < short or sen_len > long:
                print(line)
            sen_len = math.ceil(sen_len / 10) * 10
            if len_stat.get(sen_len) is not None:
                len_stat[sen_len] += 1
            else:
                len_stat[sen_len] = 1
    len_stat = list(len_stat.items())
    assert lines_cnt == sum(map(lambda x: x[1], len_stat))
    len_stat.sort(key=lambda x: x[0])
    len_stat = dict(len_stat)
    for length, cnt in len_stat.items():
        print(length, cnt)
    tools.plot_bargraph('sentence length distribution', len_stat)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-fname', type=str, metavar='STR', required=True, help='corpus filename')
    parser.add_argument('--stat-tokens', action='store_true', help='whether to show tokens stat info')
    parser.add_argument('--stat-sens', action='store_true', help='whether to show sens len stat info')
    parser.add_argument('--short', type=int, metavar='N', help='sen len < `short` will be shown')
    parser.add_argument('--long', type=int, metavar='N', help='sen len > `long` will be shown')
    parser.add_argument('--encoding', type=str, metavar='STR', help='open and save encoding')
    parser.add_argument('--debug', action='store_true', help='whether to show more info for debug')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.stat_tokens = getattr(args, 'stat_tokens', False)
    args.stat_sens = getattr(args, 'stat_sens', False)
    args.short = my_getattr(args, 'short', 2)
    args.long = my_getattr(args, 'long', 100)
    args.encoding = my_getattr(args, 'encoding', 'utf-8')
    args.debug = getattr(args, 'debug', False)

    main(args)
