"""
This script used to statistic corpus information including distribution of words and sentences.
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
    stat_words_distribution(args.corpus_fname, args.encoding)
    stat_sens_distribution(args.corpus_fname, args.encoding)


def stat_words_distribution(corpus_fname, encoding='utf-8'):
    print('=================== words distribution ========================')
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
    all_distinct_tokens = reader.read_tokens(corpus_fname, encoding)
    assert vocab_size == len(all_distinct_tokens), 'len of tcounts != corpus distinct tokens count'
    times_stat = list(times_stat.items())
    assert vocab_size == sum(map(lambda x: x[1], times_stat))
    all_tokens_num = sum(map(lambda x: x[1], tcounts))
    assert all_tokens_num == sum([times*cnt for times, cnt in times_stat])
    print('all tokens num:', all_tokens_num)
    times_stat.sort(key=lambda x: x[1], reverse=True)
    for times, cnt in times_stat:
        print('{0:6} words occur {1:10} times, the ratio of vocab_size is {2:>10.6f}% '
              '| the ratio of all tokens num is {3:>10.6f}%'
              .format(cnt, times, cnt / vocab_size * 100, cnt*times / all_tokens_num * 100))


def stat_sens_distribution(corpus_fname, encoding='utf-8'):
    line_cnt = reader.count_lines(corpus_fname)
    print('=================== sentences distribution ========================')
    len_stat = {}
    with open(corpus_fname, 'r', encoding=encoding) as file:
        for line in file:
            sen_len = len(line.split())
            if len_stat.get(sen_len) is not None:
                len_stat[sen_len] += 1
            else:
                len_stat[sen_len] = 1
    assert line_cnt == sum(map(lambda x: x[1], list(len_stat.items())))
    range_len_stat = {}
    for length, cnt in len_stat.items():
        length = math.ceil(length / 10) * 10
        if range_len_stat.get(length) is not None:
            range_len_stat[length] += cnt
        else:
            range_len_stat[length] = cnt
    assert line_cnt == sum(map(lambda x: x[1], list(range_len_stat.items())))
    tools.plot_bargraph('sentence length distribution', range_len_stat)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-fname', type=str, metavar='STR', required=True, help='corpus filename')
    parser.add_argument('--encoding', type=str, metavar='STR', help='open and save encoding')
    parser.add_argument('--debug', action='store_true', help='whether to show more info for debug')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.encoding = my_getattr(args, 'encoding', 'utf-8')
    args.debug = getattr(args, 'debug', False)

    main(args)
