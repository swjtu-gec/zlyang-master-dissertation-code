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

from utils import reader
from utils import tools
from utils.tools import my_getattr

# convert traditional Chinese to simplified Chinese
CC = OpenCC('t2s')
remove_pattern = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9、：“”‘’（）【】[\]——.-《》·。？！，；]')
empty_pattern = re.compile('\\s+')


def segment_sen(sen, char_level):
    sen = CC.convert(remove_pattern.sub(' ', sen))
    if char_level:
        segmented = tools.sen2chars(sen)
    else:
        segmented = jieba.lcut(sen)
    return list(filter(lambda x: x.strip(), segmented))


def main(args):
    assert args.nlpcc_traindata not in [args.train_all_src, args.train_all_trg], \
        "train and split files are the same"
    assert args.train_all_src != args.train_all_trg, 'src and trg filename are the same'

    with open(args.nlpcc_traindata, 'r', encoding=args.encoding) as raw_file, \
            open(args.train_all_src, 'w', encoding=args.encoding) as src_file, \
            open(args.train_all_trg, 'w', encoding=args.encoding) as trg_file:
        line_cnt = 0
        for line in raw_file:
            # remove space from nlpcc2018gec raw train data
            filed_list = list(map(lambda x: empty_pattern.sub('', x), line.split('\t')))
            filed_list = list(filter(lambda x: x, filed_list))
            orig_sen = filed_list[2]
            orig_segmented = segment_sen(orig_sen, args.char_level)
            orig_segmented_joined = ' '.join(orig_segmented) + '\n'
            if empty_pattern.match(orig_segmented_joined) or orig_segmented_joined in ['']:
                continue
            num_correct = int(filed_list[1])
            # add right => right sen pairs
            if num_correct == 0:
                src_file.write(orig_segmented_joined)
                trg_file.write(orig_segmented_joined)
            else:
                # solve the `num_correct` mismatch problem
                # '不需要修改' means: right => right sen pairs
                try:
                    for tgt_sen in filed_list[3:]:
                        if '不需要修改' in tgt_sen:
                            src_file.write(orig_segmented_joined)
                            trg_file.write(orig_segmented_joined)
                        else:
                            tgt_segmented = segment_sen(tgt_sen, args.char_level)
                            tgt_segmented_joined = ' '.join(tgt_segmented) + '\n'
                            if empty_pattern.match(tgt_segmented_joined) or tgt_segmented_joined in ['']:
                                continue
                            src_file.write(orig_segmented_joined)
                            trg_file.write(tgt_segmented_joined)
                except IndexError as error:
                    print(error.__str__())
                    print(line)
                    continue

            line_cnt += 1
            if line_cnt % 10000 == 0:
                print(line_cnt, 'lines have been processed.')

    print('=================================================')
    print(line_cnt, 'lines have been processed finally.')
    print('NLPCC 2018 GEC train data line count:', reader.count_lines(args.nlpcc_traindata, args.encoding))
    print('train all src file line count:', reader.count_lines(args.train_all_src, args.encoding))
    print('train all trg file line count:', reader.count_lines(args.train_all_trg, args.encoding))


def create_parser():
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
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.char_level = getattr(args, 'char_level', False)
    args.encoding = my_getattr(args, 'encoding', 'utf-8')
    args.debug = getattr(args, 'debug', False)

    main(args)
