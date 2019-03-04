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

from utils import reader
from utils import tools
from utils.tools import my_getattr


def main(args):
    assert args.nlpcc_traindata not in [args.train_all_src, args.train_all_trg], \
        "train and split files are the same"
    assert args.train_all_src != args.train_all_trg, 'src and trg filename are the same'

    with open(args.nlpcc_traindata, 'r', encoding=args.encoding) as raw_file, \
            open(args.train_all_src, 'w', encoding=args.encoding) as src_file, \
            open(args.train_all_trg, 'w', encoding=args.encoding) as trg_file:
        line_cnt = 0
        for line in raw_file:
            line = line.replace('\n', '').strip()
            filed_list = list(map(str.strip, line.split()))
            num_correct = int(filed_list[1])
            if num_correct + 3 != len(filed_list) or num_correct == 0:
                continue

            orig_sen = filed_list[2]
            if args.char_level:
                orig_sen_segmented = tools.sen2chars(orig_sen)
            else:
                orig_sen_segmented = jieba.lcut(orig_sen)

            for i in range(num_correct):
                tgt_sen = filed_list[3 + i]
                if args.char_level:
                    tgt_sen_segmented = tools.sen2chars(tgt_sen)
                else:
                    tgt_sen_segmented = jieba.lcut(tgt_sen)
                src_file.write(" ".join(orig_sen_segmented) + "\n")
                trg_file.write(" ".join(tgt_sen_segmented) + '\n')

            line_cnt += 1
            if line_cnt % 10000 == 0:
                print(line_cnt, 'lines have been processed.')

    print('=================================================')
    print(line_cnt, 'lines have been processed finally.')
    print('NLPCC 2018 GEC train data line count:', reader.count_lines(args.nlpcc_traindata, args.encoding))
    print('train all src file line count:', reader.count_lines(args.train_all_src, args.encoding))
    print('train all trg file line count:', reader.count_lines(args.train_all_trg, args.encoding))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlpcc-traindata', type=str, metavar='STR', required=True,
                        help='url to NLPCC 2018 GEC train data')
    parser.add_argument('--train-all-src', type=str, metavar='STR', required=True,
                        help='url to train all src data')
    parser.add_argument('--train-all-trg', type=str, metavar='STR', required=True,
                        help='url to train all trg data')
    parser.add_argument('--char-level', action='store_true', help='whether to cut sentence in char level')
    parser.add_argument('--encoding', type=str, metavar='STR', help='open and save encoding')
    parser.add_argument('--debug', action='store_true', help='whether to show more info for debug')

    args = parser.parse_args()
    args.char_level = getattr(args, 'char_level', False)
    args.encoding = my_getattr(args, 'encoding', 'utf-8')
    args.debug = getattr(args, 'debug', False)

    main(args)
