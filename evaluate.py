#!/usr/bin/env python
# coding:utf8
from __future__ import absolute_import
import copy
import codecs
from string_tool import strQ2B, drop_punctuation, filter_blank, split_string


def format_string(string):
    string = strQ2B(string)
    string = string.lower()
    string = drop_punctuation(string)
    string = filter_blank(string)
    string = string.strip()
    return string


def load_qid_answer_expand(file_path):
    qid_answer_expand = dict()
    with codecs.open(file_path, "r", 'utf8') as fin:
        for _line in fin:
            if len(_line.strip().split("\t")) != 3:
                print(_line.strip())
                continue
            qid, answer, answer_expand = _line.strip().split("\t")
            answer_expand = set(answer_expand.split("|"))
            tmp_answer_expand = copy.copy(answer_expand)
            for element in tmp_answer_expand:
                answer_expand.add(format_string(element))
            qid_answer_expand[qid] = (answer, answer_expand)
    return qid_answer_expand


def is_exact_match_answer(qid, competitor_answer, qid_answer_expand):
    if qid not in qid_answer_expand:
        raise ValueError("Invalid qid:%s" % qid)
    competitor_answer = competitor_answer.strip()
    if competitor_answer == "":
        return "0"

    format_competitor_answer = format_string(competitor_answer)
    answer, answer_expand = qid_answer_expand[qid]

    if format_competitor_answer in answer_expand:
        return "1"
    tmp_set1 = set([format_string(element) for element in drop_punctuation(competitor_answer).lower().split(" ")])
    tmp_set2 = set([format_string(element) for element in drop_punctuation(answer).lower().split(" ")])
    if tmp_set1 == tmp_set2:
        return "1"
    return "0"


def cacu_character_level_f(qid, competitor_answer, qid_answer_expand):
    if qid not in qid_answer_expand:
        raise ValueError("Invalid qid:%s" % qid)
    competitor_answer = competitor_answer.strip()
    if competitor_answer == "":
        return 0.0, 0.0, 0.0, None
    format_competitor_answer = format_string(competitor_answer)
    format_competitor_answer_tokens = split_string(format_competitor_answer)
    answer, answer_expand = qid_answer_expand[qid]
    max_f = 0.0
    max_f_precision = 0.0
    max_f_recall = 0.0
    max_f_answer = None
    for tmp_answer in answer_expand:
        tmp_answer_tokens = split_string(format_string(tmp_answer))
        tmp_answer_tokens_copy = copy.copy(tmp_answer_tokens)
        right_count = 0
        for format_competitor_answer_token in format_competitor_answer_tokens:
            if format_competitor_answer_token in tmp_answer_tokens_copy:
                right_count += 1
                tmp_answer_tokens_copy.remove(format_competitor_answer_token)
        if right_count == 0:
            continue
        precision = 1.0 * right_count / len(format_competitor_answer_tokens)
        recall = 1.0 * right_count / len(tmp_answer_tokens)
        f = 2 * precision * recall / (precision + recall)
        if f > max_f:
            max_f = f
            max_f_precision = precision
            max_f_recall = recall
            max_f_answer = tmp_answer
    return max_f, max_f_precision, max_f_recall, max_f_answer


def evalutate(predict_answer, gold_answer_file="data/qid_answer_expand/qid_answer_expand.all"):
    """
    :return:
        Query-level Precision
        Character-level Average F
    """
    qid_answer_expand = load_qid_answer_expand(gold_answer_file)
    total = 0
    right = 0
    sum_f = 0.0
    for qid in predict_answer.keys():
        try:
            competitor_answer = predict_answer[qid]
            right_flag = is_exact_match_answer(qid, competitor_answer, qid_answer_expand)
            if right_flag == "1":
                right += 1
            max_f, max_f_precision, max_f_recall, max_f_answer = cacu_character_level_f(qid, competitor_answer,
                                                                                        qid_answer_expand)
            sum_f += max_f
        except:
            print("[WARNING] Can't Evalute QID: %s" % qid)
        total += 1
    return 100. * right / total, sum_f / total * 100.
