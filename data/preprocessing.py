import json
import re

import click
import jieba_fast as jieba
from opencc import OpenCC
from smart_open import smart_open
from tqdm import tqdm

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


@click.group()
def entry_point():
    pass


entry_point.add_command(segment_wiki)


if __name__ == '__main__':
    entry_point()
