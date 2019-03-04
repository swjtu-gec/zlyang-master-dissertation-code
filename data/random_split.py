import argparse
import os
import sys

import numpy.random as rdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils import reader
from utils.tools import my_getattr

rdm.seed(1234)


def main(args):
    all_train_src = 'lang8.src'
    all_train_trg = 'lang8.trg'
    train_src = 'train.tok.src'
    train_trg = 'train.tok.trg'
    dev_src = 'dev.tok.src'
    dev_trg = 'dev.tok.trg'

    all_train_src = os.path.join(args.input_dir, all_train_src)
    all_train_trg = os.path.join(args.input_dir, all_train_trg)
    train_src = os.path.join(args.output_dir, train_src)
    train_trg = os.path.join(args.output_dir, train_trg)
    dev_src = os.path.join(args.output_dir, dev_src)
    dev_trg = os.path.join(args.output_dir, dev_trg)

    all_train_line_cnt = reader.count_lines(all_train_src, args.encoding)
    dev_ratio = args.dev_samples_num / all_train_line_cnt

    with open(all_train_src, 'r', encoding=args.encoding) as all_src_file, \
            open(all_train_trg, 'r', encoding=args.encoding) as all_trg_file, \
            open(train_src, 'w', encoding=args.encoding) as train_src_file, \
            open(train_trg, 'w', encoding=args.encoding) as train_trg_file, \
            open(dev_src, 'w', encoding=args.encoding) as dev_src_file, \
            open(dev_trg, 'w', encoding=args.encoding) as dev_trg_file:
        line_cnt = 0
        for src_line, trg_line in zip(all_src_file, all_trg_file):
            # remove empty src or target sentence pairs
            if src_line in ['', '\n'] or trg_line in ['', '\n']:
                continue
            if rdm.rand() < dev_ratio:
                dev_src_file.write(src_line)
                dev_trg_file.write(trg_line)
            else:
                train_src_file.write(src_line)
                train_trg_file.write(trg_line)
            line_cnt += 1
            if line_cnt % 10000 == 0:
                print(line_cnt, 'lines have been processed.')

    print('=================================================')
    print(line_cnt, 'lines have been processed finally.')
    print('train src data lines count:', reader.count_lines(train_src, args.encoding))
    print('train trg data lines count:', reader.count_lines(train_trg, args.encoding))
    print('dev src data lines count:', reader.count_lines(dev_src, args.encoding))
    print('dev trg data lines count:', reader.count_lines(dev_trg, args.encoding))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, metavar='STR', required=True,
                        help='dir to all src and trg data')
    parser.add_argument('--output-dir', type=str, metavar='STR', required=True,
                        help='dir to store split train and dev data')
    parser.add_argument('--dev-samples-num', type=int, metavar='N', required=True,
                        help='the number of dev data samples')
    parser.add_argument('--encoding', type=str, metavar='STR', help='open and save encoding')
    parser.add_argument('--debug', action='store_true', help='whether to show more info for debug')

    args = parser.parse_args()
    args.encoding = my_getattr(args, 'encoding', 'utf-8')
    args.debug = getattr(args, 'debug', False)

    main(args)
