import argparse
import os
import re
import sys

import jieba
from opencc import OpenCC


def do_operation_times(trg, op, times):
    for _ in range(times):
        trg = op(trg)
    return trg


PROJECT_ROOT = do_operation_times(os.path.realpath(__file__), os.path.dirname, 2)
print('project root:', PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from utils.tools import my_getattr
from utils import reader
from utils import tools

CC = OpenCC('t2s')
remove_pattern = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9、：:"“”\'‘’()（）【】[\]——.-<>《》·。?？!！,，;；]')


def segment_sen(sen, char_level, noHMM):
    sen = CC.convert(remove_pattern.sub(' ', sen))
    if char_level:
        segmented = tools.sen2chars(sen)
    else:
        if noHMM:
            segmented = jieba.lcut(sen, HMM=False)
        else:
            segmented = jieba.lcut(sen)
    return list(filter(lambda x: x.strip(), segmented))


def main(args):
    current_func_name = sys._getframe().f_code.co_name
    if args.raw_fname == args.seg_fname:
        print('\n======== In', current_func_name, '========')
        print('raw and segmented file are the same')
        print('nothing to do')
        return
    with open(args.raw_fname, 'r', encoding=args.encoding) as raw_file, \
            open(args.seg_fname, 'w', encoding=args.encoding) as seg_file:
        line_cnt = 0
        for line in raw_file:
            to_write = ' '.join(segment_sen(line, args.char_level, args.noHMM)) + '\n'
            seg_file.write(to_write)
            line_cnt += 1
            if line_cnt % 10000 == 0:
                print(line_cnt, 'lines have been processed.')

    print('=================================================')
    print(line_cnt, 'lines have been processed finally.')
    print('raw file line count:', reader.count_lines(args.raw_fname, args.encoding))
    print('segmented file line count:', reader.count_lines(args.seg_fname, args.encoding))


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-fname', type=str, metavar='STR', required=True, help='raw filename')
    parser.add_argument('--seg-fname', type=str, metavar='STR', required=True, help='segmented filename')
    parser.add_argument('--char-level', action='store_true', help='whether to cut sentence in char level')
    parser.add_argument('--noHMM', action='store_true', help='whether to cut sentence without HMM')
    parser.add_argument('--encoding', type=str, metavar='STR', help='open and save encoding')
    parser.add_argument('--debug', action='store_true', help='whether to show more info for debug')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.char_level = getattr(args, 'char_level', False)
    args.noHMM = getattr(args, 'noHMM', False)
    args.encoding = my_getattr(args, 'encoding', 'utf-8')
    args.debug = getattr(args, 'debug', False)

    main(args)
