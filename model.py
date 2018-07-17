#!/usr/bin/env python3
"""Implementation of the FusionNet reader."""

import torch
import torch.nn as nn
import fusionnet_layers as fnlayers
from layers import Embeddings
import numpy as np

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class FusionNetReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, dicts, args, feat_dicts, feat_dims):
        super(FusionNetReader, self).__init__()
        # Store config
        self.args = args
        fnlayers.set_my_dropout_prob(args['encoder_dropout'])
        fnlayers.set_seq_dropout(True)

        self.embedding = Embeddings(word_vec_size=args['word_vec_size'],
                                    dicts=dicts,
                                    feature_dicts=feat_dicts,
                                    feature_dims=feat_dims)

        self.qemb_match = fnlayers.SeqAttnMatch(args['word_vec_size'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = self.embedding.output_size + args['feature_num']
        if args['use_qemb']:
            doc_input_size += args['word_vec_size']
        question_input_size = self.embedding.output_size

        cur_doc_hidden_size = doc_input_size
        cur_question_hidden_size = question_input_size

        # Reading component
        self.reading_doc_rnn = fnlayers.RNNEncoder(input_size=cur_doc_hidden_size, hidden_size=args['hidden_size'],
                                                   num_layers=args['num_layers'],
                                                   rnn_type=self.RNN_TYPES[args['rnn_type']])
        self.reading_question_rnn = fnlayers.RNNEncoder(input_size=cur_question_hidden_size,
                                                        hidden_size=args['hidden_size'],
                                                        num_layers=args['num_layers'],
                                                        rnn_type=self.RNN_TYPES[args['rnn_type']])

        # Question understanding component
        # input: [low_level_question, high_level_question]
        # low-level + high-level

        cur_doc_hidden_size = args['hidden_size'] * 2 * args['num_layers']
        cur_question_hidden_size = args['hidden_size'] * 2 * args['num_layers']

        # Question Understanding component
        self.understanding_question_rnn = fnlayers.RNNEncoder(input_size=cur_question_hidden_size,
                                                              hidden_size=args['hidden_size'], num_layers=1,
                                                              rnn_type=self.RNN_TYPES[args['rnn_type']])
        cur_question_hidden_size = args['hidden_size'] * 2

        # Fully-Aware Multi-level Fusion
        # [word_embedding, low_level_doc_hidden, high_level_doc_hidden]
        history_of_word_size = args['word_vec_size'] + cur_doc_hidden_size
        self.full_multi_attn_doc = fnlayers.FullAttention(full_size=history_of_word_size,
                                                          hidden_size=3 * args['hidden_size'] * 2, num_level=3)

        # Multi-level rnn
        # input: [low_level_doc, high_level_doc, low_level_fusion_doc, high_level_fusion_doc,
        # understanding_level_question_fusion_doc]

        self.multi_level_rnn = fnlayers.RNNEncoder(
            input_size=cur_doc_hidden_size * 2 + cur_question_hidden_size,
            hidden_size=args['hidden_size'],
            num_layers=1,
            rnn_type=self.RNN_TYPES[args['rnn_type']]
        )

        cur_info_hidden_size = args['hidden_size'] * 2
        # Fully-Aware Self-Boosted Fusion
        history_of_doc_word_size = history_of_word_size + cur_doc_hidden_size + cur_info_hidden_size + \
                                   cur_question_hidden_size
        self.full_self_attn_doc = fnlayers.FullAttention(full_size=history_of_doc_word_size,
                                                         hidden_size=args['hidden_size'] * 2, num_level=1)

        self.self_rnn = fnlayers.RNNEncoder(
            input_size=cur_info_hidden_size * 2,
            hidden_size=args['hidden_size'],
            num_layers=1,
            rnn_type=self.RNN_TYPES[args['rnn_type']]
        )

        doc_hidden_size = 2 * args['hidden_size']
        question_hidden_size = 2 * args['hidden_size']

        # Question merging
        self.question_self_attn = fnlayers.LinearSeqAttn(question_hidden_size)

        self.start_attn = fnlayers.BilinearSeqAttn(doc_hidden_size, question_hidden_size, log_normalize=False)

        self.start_gru = nn.GRUCell(question_hidden_size, doc_hidden_size)

        self.end_attn = fnlayers.BilinearSeqAttn(doc_hidden_size, question_hidden_size)

        self.nll_loss = nn.NLLLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def score(self, batch):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1_mask = torch.eq(batch.e_text[:, :, 0], 0)
        x2_mask = torch.eq(batch.q_text, 0)

        # Embed both document and question
        e_input = batch.e_text
        x1_pos_input = self.embedding.forward(e_input)
        x1_word_emb = x1_pos_input[:, :, :self.args['word_vec_size']]
        q_input = torch.cat([batch.q_text.unsqueeze(-1), batch.q_feature], dim=-1)
        x2_input = self.embedding.forward(q_input)
        x2_word_emb = x2_input[:, :, :self.args['word_vec_size']]

        # Dropout on word embeddings
        if self.args['dropout_emb'] > 0:
            x1_word_emb = fnlayers.dropout(x1_word_emb, p=self.args['dropout_emb'], training=self.training)
            x2_word_emb = fnlayers.dropout(x2_word_emb, p=self.args['dropout_emb'], training=self.training)

        # Add manual features
        doc_input_list = [x1_pos_input]
        if self.args['feature_num'] > 0:
            doc_input_list.append(batch.e_feature)

        # Add attention-weighted question representation
        if self.args['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_word_emb, x2_word_emb, x2_mask)
            doc_input_list.append(x2_weighted_emb)

        x1_input = torch.cat(doc_input_list, -1)

        enc_doc_hiddens = self.reading_doc_rnn(x1_input, x1_mask)
        enc_question_hiddens = self.reading_question_rnn(x2_input, x2_mask)

        understanding_question_hiddens = self.understanding_question_rnn(enc_question_hiddens, x2_mask)

        history_of_doc_word = torch.cat([x1_word_emb, enc_doc_hiddens], dim=2)
        history_of_question_word = torch.cat([x2_word_emb, enc_question_hiddens], dim=2)

        fa_multi_level_doc_input1 = torch.cat([enc_doc_hiddens], dim=2)
        fa_multi_level_doc_input2 = torch.cat([enc_question_hiddens, understanding_question_hiddens], dim=2)

        fa_multi_level_doc_vectors = self.full_multi_attn_doc(history_of_doc_word, history_of_question_word,
                                                              fa_multi_level_doc_input1, fa_multi_level_doc_input2,
                                                              x2_mask)

        multi_level_doc_hiddens = self.multi_level_rnn(torch.cat([enc_doc_hiddens, fa_multi_level_doc_vectors], dim=2),
                                                       x1_mask)

        history_of_doc_word2 = torch.cat([x1_word_emb, enc_doc_hiddens, fa_multi_level_doc_vectors,
                                          multi_level_doc_hiddens], dim=2)

        self_boosted_doc_vectors = self.full_self_attn_doc(history_of_doc_word2, history_of_doc_word2,
                                                           multi_level_doc_hiddens, multi_level_doc_hiddens,
                                                           x1_mask)

        understanding_doc_hiddens = self.self_rnn(torch.cat([multi_level_doc_hiddens, self_boosted_doc_vectors], dim=2),
                                                  x1_mask)

        # shape: [batch, len_q]
        q_merge_weights = self.question_self_attn(understanding_question_hiddens, x2_mask)
        # shape: [batch, 2*hidden_size]
        question_hidden = fnlayers.weighted_avg(understanding_question_hiddens, q_merge_weights)

        # Predict start and end positions
        # shape: [batch, len_d]  SOFTMAX NOT LOG_SOFTMAX
        start_scores = self.start_attn(understanding_doc_hiddens, question_hidden, x1_mask)
        # shape: [batch, 2*hidden_size]
        gru_input = fnlayers.weighted_avg(understanding_doc_hiddens, start_scores)
        # shape: [batch, 2*hidden_size]
        memory_hidden = self.start_gru(gru_input, question_hidden)
        # shape: [batch, len_d]
        end_scores = self.end_attn(understanding_doc_hiddens, memory_hidden, x1_mask)
        # log start_scores
        if self.training:
            start_scores = torch.log(start_scores.add(1e-9))
            # end_scores = torch.log(end_scores.add(1e-9))
        return start_scores, end_scores

    def loss(self, batch):
        start_score, end_score = self.score(batch)
        start_right_score = batch.start_position
        end_right_score = batch.end_position

        start_loss = self.nll_loss(start_score, start_right_score)
        end_loss = self.nll_loss(end_score, end_right_score)

        loss = start_loss + end_loss
        return loss

    def forward(self, batch):
        return self.loss(batch)

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        """Take argmax of constrained score_s * score_e.
        from https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/model.py
        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        pred_s = []
        pred_e = []
        pred_score = []
        para_id = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
 #           scores = torch.ger(score_s[i], score_e[i])
            scores = score_s[i].unsqueeze(1) + score_e[i].unsqueeze(0)
#            scores = torch.exp(score_s[i]).unsqueeze(1) + torch.exp(score_e[i]).unsqueeze(0)

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.data.cpu().numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)

            pred_s.append(s_idx[0])  # 默认取top1，否则 改成s_idx
            pred_e.append(e_idx[0])
            pred_score.append(scores_flat[idx_sort])
            para_id.append(i)
        return pred_s, pred_e, pred_score, para_id

    def predict(self, q_evidens, top_n=1, max_len=15):
        # (batch, e_len)
        start_score, end_score = self.score(q_evidens)

        pred_s, pred_e, pred_score, para_id = self.decode(start_score, end_score, top_n=top_n, max_len=max_len)

        return pred_s, pred_e, pred_score, para_id

    @staticmethod
    def ensemble_predict(models, q_evidens, weights=None, top_n=1, max_len=15):
        fnlayers.set_my_dropout_prob(0)
        if weights is not None:
            assert len(weights) == len(models)
        else:
            weights = [1. / len(models)] * len(models)
        start_score_list, end_score_list = list(), list()
        for index, model in enumerate(models):
            start_score, end_score = model.score(q_evidens)
            start_score_list += [start_score * weights[index]]
            end_score_list += [end_score * weights[index]]
        start_score = sum(start_score_list)
        end_score = sum(end_score_list)
        pred_s, pred_e, pred_score, para_id = FusionNetReader.decode(start_score, end_score,
                                                                      top_n=top_n, max_len=max_len)
        return pred_s, pred_e, pred_score, para_id
