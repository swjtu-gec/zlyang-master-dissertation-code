import math
import os
import sys

import click
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)
from utils import reader
from utils import tools


@click.command()
@click.option('--first-file', type=str, help='the first file to compare')
@click.option('--second-file', type=str, help='the second file to compare')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def sim_two_files(first_file, second_file, encoding):
    fir_lines_cnt = reader.count_lines(first_file, encoding)
    sec_lines_cnt = reader.count_lines(second_file, encoding)
    same_count = 0
    with open(first_file, 'r', encoding=encoding) as first, \
            open(second_file, 'r', encoding=encoding) as second:
        sec_lines = set(second.read().splitlines())
        for fir_line in tqdm(first):
            if fir_line.replace('\n', '') in sec_lines:
                same_count += 1
    print('similarity:', str(same_count / fir_lines_cnt), '|', str(same_count / sec_lines_cnt))


@click.command()
@click.option('--src-fname', type=str, help='the source filename')
@click.option('--trg-fname', type=str, help='the target filename')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def len_ratio(src_fname, trg_fname, encoding):
    line_cnt = reader.count_lines(src_fname, encoding)
    assert line_cnt == reader.count_lines(trg_fname, encoding), 'line count does not match...'
    ratio_stat = {}
    with open(src_fname, 'r', encoding=encoding) as src_file, \
            open(trg_fname, 'r', encoding=encoding) as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            sen_len_ratio = math.ceil(len(trg_line.split()) / len(src_line.split()) * 10)
            if sen_len_ratio < 6 or sen_len_ratio > 16:
                print(src_line, trg_line)
            if ratio_stat.get(sen_len_ratio) is not None:
                ratio_stat[sen_len_ratio] += 1
            else:
                ratio_stat[sen_len_ratio] = 1
    ratio_stat = list(ratio_stat.items())
    assert line_cnt == sum(map(lambda x: x[1], ratio_stat))
    ratio_stat.sort(key=lambda x: x[0])
    ratio_stat = dict(ratio_stat)
    for ratio, cnt in ratio_stat.items():
        print(ratio, cnt)
    tools.plot_bargraph('trg len ratio to src len distribution', ratio_stat)


@click.group()
def entry_point():
    pass


entry_point.add_command(sim_two_files)
entry_point.add_command(len_ratio)


if __name__ == '__main__':
    entry_point()
