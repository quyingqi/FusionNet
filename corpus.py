# -*- coding:utf-8 -*-
# Author: Roger
# Created by Roger on 2017/10/24
from __future__ import absolute_import
import codecs
import math, random
try:
    import simplejson as json
except:
    import json
import torch
from torch.autograd import Variable
from layers import Constants, Dictionary


def convert2longtensor(x):
    return torch.LongTensor(x)


def convert2variable(x, device=-1, volatile=True):
    if device >= 0:
        x = x.cuda(device)
    return Variable(x, volatile=volatile)


class Evidence(object):
    def __init__(self, e_key, e_text, e_text_index, e_feature, starts, ends):
        self.e_key = e_key  # String
        self.e_text = e_text  # list(string)
        self.e_text_index = e_text_index  # torch.LongTensor
        self.e_feature = e_feature  # torch.LongTensor
        self.starts = starts  # list(int)
        self.ends = ends  # list(int)

    def __iter__(self):
        for d in [self.e_key, self.e_text, self.e_text_index, self.e_feature, self.starts, self.ends]:
            yield d

    @staticmethod
    def load_one_evidence(evidence, word_dict, pos_dict, ner_dict):
        e_key = evidence['e_key']
        e_text = evidence["evidence_tokens"]

        if 'answer_starts' in evidence:
            if len(evidence['answer_starts']) == 0:
                starts = [-1]
            else:
                starts = evidence['answer_starts']
        else:
            starts = [-1]
        if 'answer_ends' in evidence:
            if len(evidence['answer_ends']) == 0:
                ends = [-1]
            else:
                ends = evidence['answer_ends']
        else:
            ends = [-1]

        # if starts[0] == -1 or ends[0] == -1:
        #     return None

        e_text_index = convert2longtensor(word_dict.convert_to_index(e_text, Constants.UNK_WORD))

        e_pos = evidence['evidence_pos']
        e_ner = evidence['evidence_ners']
        e_ner_index = convert2longtensor(ner_dict.convert_to_index(e_ner, Constants.UNK_WORD))
        e_pos_index = convert2longtensor(pos_dict.convert_to_index(e_pos, Constants.UNK_WORD))

        qe_feature = torch.FloatTensor(evidence["qecomm"])
        ee_fre = torch.FloatTensor(evidence['fre_tokens'])
        ee_com = torch.FloatTensor(evidence['f_eecomm'])
        dis_edit = torch.FloatTensor(evidence['f_edit_dist'])
        dis_jaccard = torch.FloatTensor(evidence['f_jaccard'])
        qe_feature_c = torch.FloatTensor(evidence['qe_feature_c'])
        ee_fre_c = torch.FloatTensor(evidence['fre_token_c'])
        ee_com_c = torch.FloatTensor(evidence['f_eecomm_c'])
        dis_edit_c = torch.FloatTensor(evidence['f_edit_dist_c'])
        dis_jaccard_c = torch.FloatTensor(evidence['f_jaccard_c'])
#        ee_ratio = torch.FloatTensor(evidence['fre_ratio'])
        e_feature_index = torch.stack([e_text_index, e_pos_index, e_ner_index], dim=1)
        e_feature_float = torch.stack([qe_feature, ee_fre, ee_com, dis_edit, dis_jaccard,
                                       qe_feature_c, ee_fre_c, ee_com_c, dis_edit_c, dis_jaccard_c], dim=1)

        return Evidence(e_key, e_text, e_feature_index, e_feature_float, starts, ends)

    @staticmethod
    def batchify(data):
        e_key, e_real_text, e_feature_index, e_feature_float, starts, ends = zip(*data)
        e_feature_index_size = e_feature_index[0].size()[1]
        e_feature_float_size = e_feature_float[0].size()[1]

        e_lens = [len(e_real_text[i]) for i in range(len(data))]

        max_e_length = max(e_lens)
        e_index = e_feature_index[0].new(len(data), max_e_length, e_feature_index_size).fill_(Constants.PAD)
        e_feature = e_feature_float[0].new(len(data), max_e_length, e_feature_float_size).fill_(Constants.PAD)

        for i in range(len(data)):
            length = e_lens[i]
            e_index[i, :, :].narrow(0, 0, length).copy_(e_feature_index[i])
            e_feature[i, :, :].narrow(0, 0, length).copy_(e_feature_float[i])

        start_position = convert2longtensor([start[0] for start in starts])
        end_position = convert2longtensor([end[0] for end in ends])

        e_lens = convert2longtensor(e_lens)

        return e_index, e_feature, e_lens, start_position, end_position, e_key, e_real_text


class Question(object):
    def __init__(self, q_key, q_text, q_text_index, q_feature):
        self.q_key = q_key  # String
        self.q_text = q_text  # list(string)
        self.q_text_index = q_text_index  # torch.LongTensor
        self.q_feature = q_feature  # torch.LongTensor

    def __iter__(self):
        for d in [self.q_key, self.q_text, self.q_text_index, self.q_feature]:
            yield d

    @staticmethod
    def batchify(data):
        q_key, q_real_text, q_text_index, q_featurq_index = zip(*data)
        q_featurq_size = q_featurq_index[0].size()[1]

        q_lens = [q_text_index[i].size(0) for i in range(len(data))]
        max_q_length = max(q_lens)
        q_text = q_text_index[0].new(len(data), max_q_length).fill_(Constants.PAD)
        q_feature = q_featurq_index[0].new(len(data), max_q_length, q_featurq_size).fill_(Constants.PAD)

        for i in range(len(data)):
            length = q_text_index[i].size(0)
            q_text[i, :].narrow(0, 0, length).copy_(q_text_index[i])
            q_feature[i, :, :].narrow(0, 0, length).copy_(q_featurq_index[i])

        q_lens = convert2longtensor(q_lens)

        return q_text, q_feature, q_lens, q_key, q_real_text

    @staticmethod
    def load_one_question(data, word_dict, pos_dict, ner_dict):
        q_key = data['q_key']

        q_text = data["question_tokens"]
        q_text_index = convert2longtensor(word_dict.convert_to_index(q_text, Constants.UNK_WORD))

        q_ner = data["question_ners"]
        q_pos = data["question_pos"]
        q_ner_index = convert2longtensor(ner_dict.convert_to_index(q_ner, Constants.UNK_WORD))
        q_pos_index = convert2longtensor(pos_dict.convert_to_index(q_pos, Constants.UNK_WORD))
        q_feature = torch.stack([q_pos_index, q_ner_index], dim=1)

        return Question(q_key, q_text, q_text_index, q_feature)


class WebQACorpus(object):
    def __init__(self, filename, batch_size=64, device=-1, volatile=False,
                 word_dict=None, ner_dict=None, pos_dict=None):
        if word_dict is None:
            self.word_d, self.pos_dict, self.ner_dict = self.load_word_dictionary(filename)
        else:
            self.word_d = word_dict
            self.ner_dict = ner_dict
            self.pos_dict = pos_dict
        question_dict, evidence_dict, train_pair = self.load_data_file(filename,
                                                                       word_dict=self.word_d,
                                                                       ner_dict=self.ner_dict,
                                                                       pos_dict=self.pos_dict)
        self.question_dict = question_dict # {q_key: [question, [eid]]}
        self.evidence_dict = evidence_dict # {eid: evidence}
        self.data = train_pair  # (q_key, eid, [eid_no_answer])
        self.batch_size = batch_size
        self.device = device
        self.volatile = volatile

    def __sizeof__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def cpu(self):
        self.device = -1

    def cuda(self, device=0):
        self.device = device

    def set_device(self, device=-1):
        self.device = device

    def set_batch_size(self, batch_size=50):
        self.batch_size = batch_size

    def _question_evidence(self, question_ids, evidence_ids):
        questions = [self.question_dict[qid][0] for qid in question_ids]
        evidences = [self.evidence_dict[eid] for eid in evidence_ids]

        q_text, q_feature, q_lens, q_key, q_real_text = Question.batchify(questions)
        e_text, e_feature, e_lens, start_position, end_position, e_key, e_real_text = Evidence.batchify(evidences)

        q_text, q_feature, q_lens = [convert2variable(x, self.device, self.volatile)
                                     for x in [q_text, q_feature, q_lens]]
        e_text, e_feature, e_lens, start_position, end_position = [convert2variable(x, self.device, self.volatile)
                                                                   for x in [e_text, e_feature, e_lens,
                                                                             start_position, end_position]]

        return q_text, e_text, start_position, end_position, q_lens, e_lens, q_feature, \
               e_feature, q_key, e_key, q_real_text, e_real_text

    def next_batch(self, ranking=False, shuffle=True):
        num_batch = int(math.ceil(len(self.data) / float(self.batch_size)))

        if not shuffle:
            data = self.data
            random_indexs = torch.range(0, num_batch - 1)
        else:
            data = [self.data[index] for index in torch.randperm(len(self.data))]
            random_indexs = torch.randperm(num_batch)

        for index, i in enumerate(random_indexs):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            data_tmp = data[start:end]
            batch_qid, batch_eid, batch_negs = zip(*data_tmp)

            if ranking:
                batch_qid = list(batch_qid) * 2
                batch_eid = list(batch_eid)
                for negs in batch_negs:
                    neg = random.choice(negs)
                    batch_eid.append(neg)

            _batch_size = len(batch_qid)
            batch_data = self._question_evidence(batch_qid, batch_eid)

            q_text, e_text, start_position, end_position = batch_data[:4]
            q_lens, e_lens, q_feature, e_feature = batch_data[4:8]
            q_keys, e_keys = batch_data[8:10]

            yield Batch(q_text, e_text, start_position, end_position,
                        q_lens, e_lens, q_feature, e_feature,
                        _batch_size, q_keys, e_keys)

    def next_question(self):

        for qid in self.question_dict.keys():
            _, evidence_ids = self.question_dict[qid]
            _batch_size = len(evidence_ids)

            if _batch_size == 0:
                continue

            batch_data = self._question_evidence([qid] * _batch_size, evidence_ids)
            q_text, e_text, start_position, end_position = batch_data[:4]
            q_lens, e_lens, q_feature, e_feature = batch_data[4:8]
            q_keys, e_keys, q_real_text, e_real_text = batch_data[8:]
            yield BatchQuestion(q_text, e_text, start_position, end_position,
                                q_lens, e_lens, q_feature, e_feature,
                                _batch_size, q_keys, e_keys, e_real_text, q_real_text[0])

    @staticmethod
    def load_one_line_json(line, word_dict, pos_dict, ner_dict):
        data = json.loads(line)

        question = Question.load_one_question(data, word_dict, pos_dict, ner_dict)

        evidences = list()

        for evidence in data["evidences"]:

            evidence_data = Evidence.load_one_evidence(evidence, word_dict, pos_dict, ner_dict)

            if evidence_data is None:
                continue

            evidences.append(evidence_data)

        return question, evidences

    @staticmethod
    def load_data_file(filename, word_dict, pos_dict, ner_dict):
        question_dict = dict()
        evidence_dict = dict()
        train_pair = list()
        count = 0
        with codecs.open(filename, 'r', 'utf8') as fin:

            for line in fin:
                count += 1

                question, evidences = WebQACorpus.load_one_line_json(line, word_dict, pos_dict, ner_dict)

                all_evidence = []
                no_answer = []
                has_answer = []
                for e in evidences:
                    eid = "%s||%s" % (question.q_key, e.e_key)
                    evidence_dict[eid] = e
                    all_evidence.append(eid)

                    if e.starts[0] == -1 or e.ends[0] == -1:
                         no_answer.append(eid)
                    else:
                        has_answer.append(eid)

                question_dict[question.q_key] = [question, all_evidence]

                if count % 5000 == 0:
                    print(count)

                if not no_answer:
                    no_answer = [random.choice(list(evidence_dict.keys()))]
                if not no_answer:
                    continue
                for e in has_answer:
                    train_pair.append((question.q_key, e, no_answer))


        print('load data from %s, get %s qe pairs. ' %(filename, len(train_pair)))

        return question_dict, evidence_dict, train_pair

    @staticmethod
    def load_word_dictionary(filename, word_dict=None, pos_dict=None, ner_dict=None):
        if word_dict is None:
            word_dict = Dictionary()
            word_dict.add_specials([Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD],
                                   [Constants.PAD, Constants.UNK, Constants.BOS, Constants.EOS])
        if pos_dict is None:
            pos_dict = Dictionary()
            pos_dict.add_specials([Constants.PAD_WORD, Constants.UNK_WORD],
                                  [Constants.PAD, Constants.UNK])
        if ner_dict is None:
            ner_dict = Dictionary()
            ner_dict.add_specials([Constants.PAD_WORD, Constants.UNK_WORD],
                                  [Constants.PAD, Constants.UNK])
        with codecs.open(filename, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                for token in data["question_tokens"]:
                    word_dict.add(token)
                for pos in data['question_pos']:
                    pos_dict.add(pos)
                for ner in data['question_ners']:
                    ner_dict.add(ner)
                for evidence in data["evidences"]:
                    for token in evidence["evidence_tokens"]:
                        word_dict.add(token)
                    for pos in evidence['evidence_pos']:
                        pos_dict.add(pos)
                    for ner in evidence['evidence_ners']:
                        ner_dict.add(ner)

        return word_dict, pos_dict, ner_dict

    @staticmethod
    def load_pos_dictionary():
        return Dictionary()

    @staticmethod
    def load_ner_dictionary():
        return Dictionary()


class Batch(object):
    def __init__(self, q_text, e_text, start, end,
                 q_lens, e_lens, q_feature, e_feature,
                 batch_size, q_keys, e_keys):
        self.q_text = q_text
        self.e_text = e_text
        self.start_position = start
        self.end_position = end
        self.q_lens = q_lens
        self.e_lens = e_lens
        self.q_feature = q_feature
        self.e_feature = e_feature
        self.batch_size = batch_size
        self.pred = None
        self.q_keys = q_keys
        self.e_keys = e_keys


class BatchQuestion(object):
    def __init__(self, q_text, e_text, start, end,
                 q_lens, e_lens, q_feature, e_feature,
                 batch_size, q_keys, e_keys,
                 evidence_raw_text=None, question_raw_text=None):
        self.q_text = q_text
        self.e_text = e_text
        self.start_position = start
        self.end_position = end
        self.q_lens = q_lens
        self.e_lens = e_lens
        self.q_feature = q_feature
        self.e_feature = e_feature
        self.batch_size = batch_size
        self.pred = None
        self.q_keys = q_keys
        self.e_keys = e_keys
        self.evidence_raw_text = evidence_raw_text
        self.question_raw_text = question_raw_text


def test():
    corpus = WebQACorpus("data/baidu_data.json")
    for data in corpus.next_question():
        for index, (start, end, leng) in enumerate(torch.cat([data.start_position.unsqueeze(-1),
                                                              data.end_position.unsqueeze(-1),
                                                              data.e_lens.unsqueeze(-1)],
                                                             1)):
            print(''.join(data.evidence_raw_text[index][start.data[0]:end.data[0] + 1]))


if __name__ == "__main__":
    test()
