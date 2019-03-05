import argparse
import os
import sys

import jieba


def do_operation_times(trg, op, times):
    for _ in range(times):
        trg = op(trg)
    return trg


PROJECT_ROOT = do_operation_times(os.path.realpath(__file__), os.path.dirname, 2)
print('project root:', PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from utils.tools import my_getattr
from utils import reader


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
            seg_file.write(" ".join(jieba.lcut(line)))
            line_cnt += 1
            if line_cnt % 10000 == 0:
                print(line_cnt, 'lines have been processed.')

    print('=================================================')
    print(line_cnt, 'lines have been processed finally.')
    print('raw file line count:', reader.count_lines(args.raw_fname, args.encoding))
    print('segmented file line count:', reader.count_lines(args.seg_fname, args.encoding))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-fname', type=str, metavar='STR', required=True, help='raw filename')
    parser.add_argument('--seg-fname', type=str, metavar='STR', required=True, help='segmented filename')
    parser.add_argument('--encoding', type=str, metavar='STR', help='open and save encoding')
    parser.add_argument('--debug', action='store_true', help='whether to show more info for debug')

    args = parser.parse_args()
    args.encoding = my_getattr(args, 'encoding', 'utf-8')
    args.debug = getattr(args, 'debug', False)

    main(args)
