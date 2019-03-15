import json
import math
import os
import re
import sys

import click
import jieba_fast as jieba
from opencc import OpenCC
from smart_open import smart_open
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)

from utils import reader

jieba.initialize()

# convert traditional Chinese to simplified Chinese
CC = OpenCC('t2s')
# remove non Chinese char and so on.
REGEX = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9、：“”‘’（）【】[\]——.-《》·。？！，；]')


def segment_line(line):
    line = CC.convert(REGEX.sub(' ', line))
    return list(filter(lambda x: x.strip(), jieba.cut(line)))


def _wrap_func(to_process):
    seg_tmp = ' '.join(segment_line(to_process))
    if seg_tmp not in ['', '\n']:
        seg_tmp += '\n'
        return seg_tmp.encode('utf-8')
    else:
        return ''.encode('utf8')


@click.command()
@click.option('--input-file', type=str, help='url to wiki.json.gz')
@click.option('--output-file', type=str, help='url to wiki.seg.txt')
def segment_wiki(input_file, output_file):
    """
    :return: nothing to return
    """
    with smart_open(input_file) as fin:
        with smart_open(output_file, 'wb') as fout:
            for line in tqdm(fin):
                article = json.loads(line)
                fout.write(_wrap_func(article['title']))
                for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                    fout.write(_wrap_func(section_title))
                    for text in section_text.splitlines():
                        fout.write(_wrap_func(text))


@click.command()
@click.option('--raw-fname', type=str, help='the raw filename')
@click.option('--after-fname', type=str, help='after removed filename')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def remove_same(raw_fname, after_fname, encoding):
    assert raw_fname != after_fname, 'raw and trg filename are the same'
    with open(raw_fname, 'r', encoding=encoding) as raw_file, \
            open(after_fname, 'w', encoding=encoding) as after_file:
        distinct = set(raw_file.read().splitlines())
        for line in distinct:
            after_file.write(line + '\n')
    print('=================================================')
    raw_lines_cnt = reader.count_lines(raw_fname, encoding)
    after_lines_cnt = reader.count_lines(after_fname, encoding)
    print('raw filename:', raw_lines_cnt)
    print('after filename:', after_lines_cnt)
    print('remove %.2f%% data' % ((raw_lines_cnt - after_lines_cnt)/raw_lines_cnt*100))


@click.command()
@click.option('--src-fname', type=str, help='the source filename')
@click.option('--trg-fname', type=str, help='the target filename')
@click.option('--low', type=int, help='len ratio < `low` will be removed')
@click.option('--high', type=int, help='len ratio > `high` will be removed')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def remove_len_ratio(src_fname, trg_fname, low, high, encoding):
    suffix = '.remove.lenratio'
    raw_lines_cnt = reader.count_lines(src_fname, encoding)
    assert raw_lines_cnt == reader.count_lines(trg_fname, encoding), 'line count does not match...'
    with open(src_fname, 'r', encoding=encoding) as src_file, \
            open(trg_fname, 'r', encoding=encoding) as trg_file, \
            open(src_fname+suffix, 'w', encoding=encoding) as src_remove, \
            open(trg_fname+suffix, 'w', encoding=encoding) as trg_remove:
        for src_line, trg_line in zip(src_file, trg_file):
            sen_len_ratio = math.ceil(len(trg_line.split()) / len(src_line.split()) * 10)
            if sen_len_ratio < low or sen_len_ratio > high:
                continue
            else:
                src_remove.write(src_line)
                trg_remove.write(trg_line)
    print('=================================================')
    after_lines_cnt = reader.count_lines(src_fname+suffix, encoding)
    assert after_lines_cnt == reader.count_lines(trg_fname+suffix, encoding), 'line count does not match...'
    print('before lines count:', raw_lines_cnt)
    print('after remove:', after_lines_cnt)
    print('remove %.2f%% data' % ((raw_lines_cnt - after_lines_cnt) / raw_lines_cnt * 100))


@click.command()
@click.option('--src-fname', type=str, help='the source filename')
@click.option('--trg-fname', type=str, help='the target filename')
@click.option('--short', type=int, help='sen pairs len < `short` will be removed')
@click.option('--long', type=int, help='sen pairs len > `long` will be removed')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def remove_long_short(src_fname, trg_fname, short, long, encoding):
    suffix = '.remove.longshort'
    raw_lines_cnt = reader.count_lines(src_fname, encoding)
    assert raw_lines_cnt == reader.count_lines(trg_fname, encoding), 'line count does not match...'
    with open(src_fname, 'r', encoding=encoding) as src_file, \
            open(trg_fname, 'r', encoding=encoding) as trg_file, \
            open(src_fname+suffix, 'w', encoding=encoding) as src_remove, \
            open(trg_fname+suffix, 'w', encoding=encoding) as trg_remove:
        for src_line, trg_line in zip(src_file, trg_file):
            if short <= len(src_line.split()) <= long and short <= len(trg_line.split()) <= long:
                src_remove.write(src_line)
                trg_remove.write(trg_line)
    print('=================================================')
    after_lines_cnt = reader.count_lines(src_fname+suffix, encoding)
    assert after_lines_cnt == reader.count_lines(trg_fname+suffix, encoding), 'line count does not match...'
    print('before lines count:', raw_lines_cnt)
    print('after remove:', after_lines_cnt)
    print('remove %.2f%% data' % ((raw_lines_cnt - after_lines_cnt) / raw_lines_cnt * 100))


@click.group()
def entry_point():
    pass


entry_point.add_command(segment_wiki)
entry_point.add_command(remove_same)
entry_point.add_command(remove_len_ratio)
entry_point.add_command(remove_long_short)


if __name__ == '__main__':
    entry_point()
