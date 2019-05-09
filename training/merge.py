import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils.tools import my_getattr


def main(args):
    merge(args.to_merge_files, args.trg_fname, args.encoding)


def merge(to_merge_files, trg_fname, encoding='utf-8'):
    to_merge_fnames = to_merge_files.split()
    merged = {}
    for fname in to_merge_fnames:
        with open(fname, 'r', encoding=encoding) as file:
            for line in file:
                try:
                    cur_id = int(line.split('|||')[0].strip())
                except ValueError as error:
                    print(error)
                    continue
                if cur_id not in merged:
                    merged[cur_id] = [line]
                else:
                    merged[cur_id].append(line)

    with open(trg_fname, 'w', encoding=encoding) as trg_file:
        merged = list(merged.items())
        merged.sort(key=lambda x: x[0])
        for cur_id, sens in merged:
            for sen in sens:
                trg_file.write(sen)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--to-merge-files', type=str, metavar='STR', required=True,
                        help='to merge these reformatted files')
    parser.add_argument('--trg-fname', type=str, metavar='STR', required=True, help='merged filename')
    parser.add_argument('--encoding', type=str, metavar='STR', help='open and save encoding')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.encoding = my_getattr(args, 'encoding', 'utf-8')

    main(args)
