import os
import sys

import matplotlib.pyplot as plt


def get_fnames_under_path(path):
    """
    get filename seq under path.
    :param path: string
    :return: filename seq
    """
    if not os.path.isdir(path):
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         '() function, path type error.')
    fnames = set()
    for fname in os.listdir(path):
        fname = os.path.join(path, fname)
        if os.path.isdir(fname):
            continue
        fnames.add(fname)
    return fnames


def del_file_under_path(path):
    if not os.path.isdir(path):
        raise ValueError('In ' + sys._getframe().f_code.co_name +
                         '() function, path type error.')
    for fname in os.listdir(path):
        fname = os.path.join(path, fname)
        if os.path.isdir(fname):
            continue
        os.remove(fname)


def my_getattr(obj, attr, default):
    """
    if not obj.attr:
        return default
    else:
        return obj.attr
    """
    return default if not obj.__getattribute__(attr) else obj.__getattribute__(attr)


def sen2chars(sen, is_latin=False):
    """
    Convert sentence to char sequence.
    :param sen: str, like '今天天气很好'
    :param is_latin: whether to transform whitespace to underline.
    :return: char list, like ['今', '天', '天', '气', '很', '好']
    """
    if is_latin:
        sen = sen.replace(' ', '_')
    return [char for char in sen if not char.isspace()]


def plot_bargraph(fig_name, to_draw_dict):
    name_list = []
    num_list = []
    for i in to_draw_dict.keys():
        name_list.append(i)
        num_list.append(to_draw_dict[i])
    rects = plt.bar(range(len(num_list)), num_list, color='rgby')
    plt.ylabel('Nums')
    index = [i for i in range(len(num_list))]
    plt.xticks(index, name_list)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2, height, str(height), ha='center', va='bottom')
    plt.savefig('./' + fig_name + '.png')
    plt.show()
