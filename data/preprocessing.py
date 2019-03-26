import json
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
from utils import tools

# convert traditional Chinese to simplified Chinese
CC = OpenCC('t2s')
# remove non Chinese char and so on.
REGEX = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9、：“”‘’（）【】[\]——.-《》·。？！，；]')
remove_pattern = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9、：:"“”\'‘’()（）【】[\]——.-<>《》·。?？!！,，;；]')
empty_pattern = re.compile('\\s+')


def segment_sen(sen, char_level):
    sen = CC.convert(remove_pattern.sub(' ', sen))
    if char_level:
        segmented = tools.sen2chars(sen)
    else:
        segmented = jieba.lcut(sen)
    return list(filter(lambda x: x.strip(), segmented))


def segment_line(line, char_level):
    line = CC.convert(REGEX.sub(' ', line))
    if char_level:
        segmented = tools.sen2chars(line)
    else:
        segmented = jieba.lcut(line)
    return list(filter(lambda x: x.strip(), segmented))


def _wrap_func(to_process, char_level):
    seg_tmp = ' '.join(segment_line(to_process, char_level))
    if seg_tmp not in ['', '\n']:
        seg_tmp += '\n'
        return seg_tmp.encode('utf-8')
    else:
        return ''.encode('utf8')


@click.command()
@click.option('--input-file', type=str, help='url to wiki.json.gz')
@click.option('--output-file', type=str, help='url to wiki.seg.txt')
@click.option('--char-level', type=bool, help='whether to cut sentence in char level')
def segment_wiki(input_file, output_file, char_level):
    """
    :return: nothing to return
    """
    assert input_file != output_file, 'input and output filenames are the same'
    with smart_open(input_file) as fin:
        with smart_open(output_file, 'wb') as fout:
            for line in tqdm(fin):
                article = json.loads(line)
                fout.write(_wrap_func(article['title'], char_level))
                for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                    fout.write(_wrap_func(section_title, char_level))
                    for text in section_text.splitlines():
                        fout.write(_wrap_func(text, char_level))


@click.command()
@click.option('--src-fname', type=str, help='the source filename')
@click.option('--trg-fname', type=str, help='the target filename')
@click.option('--src-seg', type=str, help='the src segmented filename')
@click.option('--trg-seg', type=str, help='the trg segmented filename')
@click.option('--char-level', type=bool, help='whether to cut sentence in char level')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def segment_src_trg(src_fname, trg_fname, src_seg, trg_seg, char_level, encoding):
    assert src_fname != src_seg, 'source raw and segmented filename are the same'
    assert trg_fname != trg_seg, 'target raw and segmented filename are the same'
    with open(src_fname, 'r', encoding=encoding) as src_file, \
            open(trg_fname, 'r', encoding=encoding) as trg_file, \
            open(src_seg, 'w', encoding=encoding) as src_seg_file, \
            open(trg_seg, 'w', encoding=encoding) as trg_seg_file:
        lines_cnt = 0
        for src_line, trg_line in zip(src_file, trg_file):
            src_to_write = ' '.join(segment_sen(src_line, char_level)) + '\n'
            trg_to_write = ' '.join(segment_sen(trg_line, char_level)) + '\n'
            if empty_pattern.match(src_to_write) or src_to_write in ['']:
                continue
            if empty_pattern.match(trg_to_write) or trg_to_write in ['']:
                continue
            src_seg_file.write(src_to_write)
            trg_seg_file.write(trg_to_write)
            lines_cnt += 1
            if lines_cnt % 100000 == 0:
                print(lines_cnt, 'lines have been processed.')

    print('=================================================')
    print(lines_cnt, 'lines have been processed finally.')
    print('raw file lines count:', reader.count_lines(src_fname, encoding))
    segmented_lines_cnt = reader.count_lines(src_seg, encoding)
    assert segmented_lines_cnt == reader.count_lines(trg_seg, encoding), 'segmented lines count does not match...'
    print('segmented file lines count:', segmented_lines_cnt)


@click.command()
@click.option('--raw-fname', type=str, help='the raw filename')
@click.option('--after-fname', type=str, help='after removed filename')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def remove_same(raw_fname, after_fname, encoding):
    assert raw_fname != after_fname, 'raw and trg filename are the same'
    with open(raw_fname, 'r', encoding=encoding) as raw_file, \
            open(after_fname, 'w', encoding=encoding) as after_file:
        distinct = set()
        for line in raw_file:
            if line not in distinct:
                distinct.add(line)
                after_file.write(line)
    print('=================================================')
    raw_lines_cnt = reader.count_lines(raw_fname, encoding)
    after_lines_cnt = reader.count_lines(after_fname, encoding)
    print('before lines count:', raw_lines_cnt)
    print('after remove:', after_lines_cnt)
    print('remove %.2f%% data' % ((raw_lines_cnt - after_lines_cnt)/raw_lines_cnt*100))


@click.command()
@click.option('--src-fname', type=str, help='the source filename')
@click.option('--trg-fname', type=str, help='the target filename')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def remove_same_src_trg(src_fname, trg_fname, encoding):
    raw_lines_cnt = reader.count_lines(src_fname, encoding)
    assert raw_lines_cnt == reader.count_lines(trg_fname, encoding), 'lines count does not match...'
    src_after = src_fname+'.removesame'
    trg_after = trg_fname+'.removesame'
    with open(src_fname, 'r', encoding=encoding) as src_file, \
            open(trg_fname, 'r', encoding=encoding) as trg_file, \
            open(src_after, 'w', encoding=encoding) as src_after_file, \
            open(trg_after, 'w', encoding=encoding) as trg_after_file:
        distinct = set()
        for src_line, trg_line in zip(src_file, trg_file):
            if src_line+trg_line not in distinct:
                distinct.add(src_line+trg_line)
                src_after_file.write(src_line)
                trg_after_file.write(trg_line)
    print('=================================================')
    after_lines_cnt = reader.count_lines(src_after, encoding)
    assert after_lines_cnt == reader.count_lines(trg_after, encoding), 'lines count does not match...'
    print('before lines count:', raw_lines_cnt)
    print('after remove:', after_lines_cnt)
    print('remove %.2f%% data' % ((raw_lines_cnt - after_lines_cnt) / raw_lines_cnt * 100))


@click.command()
@click.option('--src-fname', type=str, help='the source filename')
@click.option('--trg-fname', type=str, help='the target filename')
@click.option('--low', type=float, help='len ratio < `low` will be removed, e.g: 0.1')
@click.option('--high', type=float, help='len ratio > `high` will be removed, e.g: 9')
@click.option('--encoding', type=str, default='utf-8', help='open and save encoding')
def remove_len_ratio(src_fname, trg_fname, low, high, encoding):
    suffix = '.remove_low'+str(low)+'_high'+str(high)
    raw_lines_cnt = reader.count_lines(src_fname, encoding)
    assert raw_lines_cnt == reader.count_lines(trg_fname, encoding), 'line count does not match...'
    with open(src_fname, 'r', encoding=encoding) as src_file, \
            open(trg_fname, 'r', encoding=encoding) as trg_file, \
            open(src_fname+suffix, 'w', encoding=encoding) as src_remove, \
            open(trg_fname+suffix, 'w', encoding=encoding) as trg_remove:
        for src_line, trg_line in zip(src_file, trg_file):
            sen_len_ratio = len(trg_line.split()) / len(src_line.split())
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
    suffix = '.remove_short'+str(short)+'_long'+str(long)
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
entry_point.add_command(segment_src_trg)
entry_point.add_command(remove_same)
entry_point.add_command(remove_same_src_trg)
entry_point.add_command(remove_len_ratio)
entry_point.add_command(remove_long_short)


if __name__ == '__main__':
    entry_point()
