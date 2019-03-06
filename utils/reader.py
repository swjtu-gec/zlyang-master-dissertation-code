import os
import re
import sys

from utils import tools

match_newline_pattern = re.compile('\n+')


def generate_text_from_corpus(path, open_encoding='utf-8'):
    """
    生成器函数，一次返回一个文本的全部内容
    :param path: corpus path
    :param open_encoding: open file encoding
    :return: 返回迭代器，可以遍历path下所有文件的内容
    """
    if not os.path.isdir(path):
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         ' func, argument should be path.')
    fnames = tools.get_fnames_under_path(path)
    for fname in fnames:
        with open(fname, 'r', encoding=open_encoding) as file:
            yield file.read()


def count_lines(url, open_encoding='utf-8'):
    line_count = 0
    if os.path.isdir(url):
        for text in generate_text_from_corpus(url, open_encoding):
            for line in match_newline_pattern.split(text):
                if line == '':
                    continue
                line_count += 1
    else:
        with open(url, 'r', encoding=open_encoding) as file:
            for line in file:
                if line != '\n' and line != '':
                    line_count += 1
    return line_count


def read_tokens(url, open_encoding='utf-8'):
    """
    Read all distinct tokens.
    :param url:
    :param open_encoding:
    :return: set, {'apple', 'banana', ...}
    """
    ret_tokens = set()
    if os.path.isdir(url):
        for text in generate_text_from_corpus(url, open_encoding):
            for line in match_newline_pattern.split(text):
                for token in line.split():
                    ret_tokens.add(token)
    elif os.path.isfile(url):
        with open(url, 'r', encoding=open_encoding) as file:
            for line in file:
                for token in line.split():
                    ret_tokens.add(token)
    return ret_tokens
