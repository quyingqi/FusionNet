#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/28
from __future__ import absolute_import
import torch
from utils import add_argument
from corpus import WebQACorpus


def preprocess_data(args):
    '''
    w, p, n = WebQACorpus.load_word_dictionary(args.baidu_file)
    word_dict, pos_dict, ner_dict = WebQACorpus.load_word_dictionary(args.train_file, w, p, n)
    word_dict.cut_by_top(args.topk)
    torch.save([word_dict, pos_dict, ner_dict], open(args.dict_file, 'wb'))
    '''
    print(args.baidu_data)
    print(args.train_data)
    print(args.valid_data)
    word_dict, pos_dict, ner_dict = torch.load(args.dict_file)

#    baidu_data = WebQACorpus(args.baidu_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
#    print("saving baidu_data ...")
#    with open(args.baidu_data, 'wb') as output:
#        torch.save(baidu_data, output)

    train_data = WebQACorpus(args.train_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
    print("saving train_data ...")
    with open(args.train_data, 'wb') as output:
        torch.save(train_data, output)

    valid_data = WebQACorpus(args.valid_file, word_dict=word_dict, pos_dict=pos_dict, ner_dict=ner_dict)
    print("saving valid_data ...")
    with open(args.valid_data, 'wb') as output:
        torch.save(valid_data, output)

if __name__ == "__main__":
    args = add_argument()
    preprocess_data(args)
