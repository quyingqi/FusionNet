# -*- coding:utf-8 -*-
from __future__ import absolute_import
from builtins import chr
import re
import codecs


blank_regexp = re.compile(r'\s+')
punctuation = set()

with codecs.open("data/qid_answer_expand/punctuation", "r", 'utf8') as fin:
    for line in fin:
        punctuation.add(line.strip())


def drop_punctuation(string):
    """删除所有标点符号"""
    rstring = ""
    for uchar in string:
        if uchar not in punctuation:
            rstring += uchar
        else:
            rstring += " "
    return rstring


def split_string(string):
    split_tokens = []
    for uchar in string:
        split_tokens.append(uchar)
    return split_tokens


def strQ2B(string):
    """全角转半角"""
    rstring = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def strB2Q(string):
    """半角转全角"""
    rstring = ""
    for uchar in string:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring


def filter_blank(string):
    return blank_regexp.sub('', string)


def filter_extra_blank(string):
    return blank_regexp.sub(' ', string)
