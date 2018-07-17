# -*- coding:utf-8 -*- 
# Author: Roger
import math
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from .mask_util import lengths2mask


class SoftmaxClassifier(nn.Module):
    """
    Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    Softmax is defined as
    :math:`f_i(x) = exp(x_i) / sum_j exp(x_j)`

    Shape:
        - Input: :math:`(N, L)`
        - Output: :math:`(N, L)`
    """

    def __init__(self):
        super(SoftmaxClassifier, self).__init__()

        self.classifier = nn.Softmax()

    def forward(self, x):
        """
        :param x: `(N, L)`
        :return: `(N, L)`
        """
        assert x.dim() == 2, 'Softmax requires a 2D tensor as input'
        return self.classifier.forward(x)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

    def predict_prob(self, x):
        """
        :param x: `(N, L)`
        :return: `(N, L)`
        """
        return self.forward(x)

    @staticmethod
    def prob2label(self, prob):
        """
        :param prob: `(N, L)`
        :return: `(N, L)`
        """
        _, pred_label = torch.max(prob, 1)
        return pred_label

    def predict(self, x, prob=True):
        return self.predict_prob(x) if prob else self.predict_label(x)


class MLPClassifer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout=0):
        super(MLPClassifer, self).__init__()

        out_component = OrderedDict()
        out_component['dropout'] = nn.Dropout(dropout)
        out_component['linear'] = nn.Linear(input_dim, output_dim)

        self.encoder = nn.Sequential(out_component)

    def forward(self, x):
        return self.encoder.forward(x)


def log_sum_exp(x, dim=0):
    """
    :param x: (batch, label)
    :return:
    """
    max_value, _ = torch.max(x, dim)
    max_exp = max_value.unsqueeze(dim).expand_as(x)
    return max_value + torch.log(torch.sum(torch.exp(x - max_exp), dim))


class CRFClassifier(nn.Module):
    def __init__(self, label_num):
        super(CRFClassifier, self).__init__()

        self.label_num = label_num
        self.start_idx = 0
        self.stop_idx = label_num - 1

        # BOS, label_num, EOS
        # to, from
        self.transit_matrix = Parameter(torch.Tensor(self.label_num, self.label_num))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.label_num)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def score_all_seqs(self, x, lengths=None):
        """
        :param x: (batch, length, label)
        :param lengths: (batch, )
        :return: log(sum(exp(all)))
        """
        mask = None

        # Check Size
        batch_size, max_len, label_num = x.size()
        if lengths is not None:
            assert batch_size == lengths.size()[0]
            mask = lengths2mask(lengths, max_len)
        assert label_num == self.label_num

        # Each Time Save Label Size
        # (Batch, Label)
        forward_var = x.data.new(batch_size, self.label_num).fill_(-10000)
        # Start at BOS
        forward_var[:, self.start_idx] = 0
        forward_var = Variable(forward_var)

        # 0, 1, ..., max_len - 1
        for i in range(max_len):

            '''
            # inefficiency
            
            alphas_t = []  # The forward variables at this timestep
            
            for next_tag in range(self.label_num):
                # (label, ) -> (batch, label)
                trans_score = self.transit_matrix[next_tag].expand(batch_size, self.label_num)

                # (batch, ) -> (batch, label)
                feature_tag = x[:, i, next_tag].unsqueeze(1).expand(batch_size, self.label_num)

                # (batch, label) + (batch, label) + (batch, label) -> (batch, label)
                next_tag_var = forward_var + trans_score + feature_tag

                # (batch, label) -> (batch, )
                alphas_t.append(log_sum_exp(next_tag_var, 1))

            forward_var_next = torch.stack(alphas_t, 1)
            '''


            # (from, to) -> (batch, from, to)
            transit_matrix = self.transit_matrix.unsqueeze(0).expand(batch_size, self.label_num, self.label_num)

            # (batch, from) -> (batch, from, from)
            pre_tag_var = forward_var.unsqueeze(1).expand(batch_size, self.label_num, self.label_num)

            # (batch, to) -> (batch, to, to)
            score = x[:, i].unsqueeze(-1).expand(batch_size, self.label_num, self.label_num)

            # (batch, label_num, label_num) -> (batch, label_num)
            forward_var_next = transit_matrix + pre_tag_var + score
            forward_var_next = log_sum_exp(forward_var_next, 2)

            if mask is not None:
                forward_var = mask[:, i][:, None] * forward_var_next + (1 - mask[:, i])[:, None] * forward_var
            else:
                forward_var = forward_var_next

        terminal_var = forward_var + self.transit_matrix[self.stop_idx].unsqueeze(0).expand_as(forward_var)
        alpha = log_sum_exp(terminal_var, 1)
        return alpha

    def viterbi_decode(self, x, lengths=None):
        """
        :param x: (length, len, label)
        :return:
        """

        mask = None

        def argmax(vec):
            # return the argmax as a python int
            _, idx = torch.max(vec, 1)
            return idx

        # Check Size
        batch_size, max_len, label_num = x.size()
        assert label_num == self.label_num
        init_vars = x.data.new(batch_size, self.label_num).fill_(-1000.)
        init_vars[:, self.start_idx] = 0

        if lengths is not None:
            mask = lengths2mask(lengths, max_len)

        # (batch, label_num)
        forward_var = Variable(init_vars)
        backpointers = list()

        for i, feat in enumerate(torch.split(x, 1, dim=1)):
            # feat (batch, label_num)
            feat = feat.squeeze(1)
            bptrs_t = []
            viterbivars_t = []
            m = mask[:, i]

            for next_tag in range(self.label_num):
                # (batch, label_num)
                trans_score = self.transit_matrix[next_tag].expand(batch_size, self.label_num)
                next_tag_var = forward_var + trans_score
                # 利用Mask，控制加的次数
                if mask is not None:
                    next_tag_var = m[:, None] * next_tag_var + (1 - m)[:, None] * forward_var
                # (batch, label_num) -> (batch)
                viterbivars_tag, best_tag_ids = torch.max(next_tag_var, 1)
                # push (batch, )
                bptrs_t += [best_tag_ids]
                # push (batch, )
                viterbivars_t += [viterbivars_tag]

            forward_var_next = torch.stack(viterbivars_t, 1) + feat

            if mask is not None:
                forward_var = m[:, None] * forward_var_next + (1 - m)[:, None] * forward_var
            else:
                forward_var = forward_var_next
            # print forward_var

            bptrs_t = torch.stack(bptrs_t, 1)
            backpointers.append(bptrs_t)
        forward_var += self.transit_matrix[self.stop_idx].expand(batch_size, self.label_num)
        scores, idx = forward_var.max(1)
        paths = [idx]

        for argmax, i in zip(reversed(backpointers), reversed(range(mask.size(1)))):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = mask[:, i].long()[:, None] * idx + (1 - mask[:, i])[:, None].long() * idx_exp
            idx = idx.squeeze(-1)
            paths.insert(0, idx.unsqueeze(1))

        paths = torch.stack(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def _viterbi_decode(self, x):
        """
        :param x: (length, label)
        :return:
        """
        backpointers = []

        def to_scalar(var):
            # returns a python float
            return var.view(-1).data.tolist()[0]

        def argmax(vec):
            # return the argmax as a python int
            _, idx = torch.max(vec, 1)
            return to_scalar(idx)

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.label_num).fill_(-10000.)
        init_vvars[0][0] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = Variable(init_vvars)
        for feat in x:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.label_num):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transit_matrix[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transit_matrix[self.stop_idx]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.start_idx  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def transition_score_given_seq(self, tag_seq, lengths=None):
        """
        :param tag_seq: (batch, len)
        :param lengths: (batch)
        :return:
        """
        # Check Size
        batch_size, max_len = tag_seq.size()
        if isinstance(tag_seq, Variable):
            pad_tag_seq = Variable(tag_seq.data.new(batch_size, max_len + 2).fill_(self.stop_idx))
        else:
            pad_tag_seq = Variable(tag_seq.new(batch_size, max_len + 2).fill_(self.stop_idx))

        pad_tag_seq[:, 0] = self.start_idx
        pad_tag_seq[:, -1] = self.stop_idx

        # TODO When PyTorch Fix its Bug that LongTensor has no neg() in Conda Release, I will Optimize this Code
        if lengths is not None:
            if not isinstance(tag_seq, Variable):
                tag_seq_v = Variable(tag_seq)
            else:
                tag_seq_v = tag_seq
            mask = lengths2mask(lengths, max_len)
            pad_tag_seq[:, 1:-1] = tag_seq_v * mask.long() + pad_tag_seq[:, 1:-1] * (1 - mask).long()
        else:
            pad_tag_seq[:, 1:-1] = tag_seq

        # print(tag_seq)
        # print(pad_tag_seq)
        # (label, label) -> (batch, label, label)
        # to, from -> batch, to, from
        trn = self.transit_matrix.unsqueeze(0).expand(batch_size, self.label_num, self.label_num)

        # reduce `to`, left all `from`
        to_index = pad_tag_seq[:, 1:]
        to_index = to_index.unsqueeze(-1).expand(batch_size, max_len + 1, self.label_num)
        # gather: trn       (input)  (batch, to,  from)
        # gather: to_index  (index)  (batch, len + 1, from)
        # gather: trn_to    (otuput) (batch, len + 1, from)
        trn_to = torch.gather(trn, 1, to_index)

        # reduce `from`, left (batch, len + 1)
        # (batch, len + 1, 1)
        from_index = pad_tag_seq[:, :-1].unsqueeze(-1)
        # gather: trn_to     (input)  (batch, len + 1, from)
        # gather: from_index (index)  (batch, len + 1, 1)
        # gather: trn_to     (otuput) (batch, len + 1, 1)
        trn_scr = torch.gather(trn_to, 2, from_index)
        trn_scr = trn_scr.squeeze(-1)

        if lengths is not None:
            mask = lengths2mask(lengths + 1, max_len + 1)
            trn_scr = trn_scr * mask

        return trn_scr.sum(1)

    def seq_score_given_seq(self, seq_score, tag_seq, lengths=None):
        """
        :param seq_score: (batch, len, label)
        :param tag_seq:   (batch, len)
        :param lengths:   (batch)
        :return:
        """
        # Check Size
        batch_size, max_len, label_num = seq_score.size()
        assert batch_size == tag_seq.size()[0]
        assert max_len == tag_seq.size()[1]
        if lengths is not None:
            assert lengths.size(0) == batch_size

        seq_label_score = torch.gather(seq_score, 2, tag_seq.unsqueeze(-1))
        seq_label_score = seq_label_score.squeeze(-1)

        if lengths is not None:
            mask = lengths2mask(lengths, max_len)
            seq_label_score = mask * seq_label_score
        return seq_label_score.sum(1)

    def score_given_seq(self, seq_score, tag_seq, lengths=None):
        transition_score = self.transition_score_given_seq(tag_seq, lengths)
        seq_tag_score = self.seq_score_given_seq(seq_score, tag_seq, lengths)
        return transition_score + seq_tag_score

    def neg_log_loss(self, x, y, lengths=None, size_average=True):
        """
        :param x: (batch, len, label)
        :param y: (batch, len)
        :param lengths: (batch)
        :return: (batch)
        """
        score_right = self.score_given_seq(seq_score=x, tag_seq=y, lengths=lengths)
        log_sum_exp_all_score = self.score_all_seqs(x, lengths=lengths)
        # print(log_sum_exp_all_score.data, score_right.data)
        if size_average:
            return torch.mean(log_sum_exp_all_score - score_right)
        else:
            return log_sum_exp_all_score - score_right

    def _score_given_seq(self, seq_score, tag_seq, lengths=None):
        """
        :param seq_score: (batch, len, label)
        :param tag_seq: (batch, len)
        :return:
        """
        # Check Sizes
        batch_size, max_len, label_num = seq_score.size()
        assert batch_size == tag_seq.size()[0]
        assert max_len == tag_seq.size(1)
        if lengths is not None:
            assert batch_size == lengths.size()[0]

        scores = list()

        for i in range(batch_size):
            scores += [self._score_sentence(seq_score[i][:lengths[i].data[0]], tag_seq[i][:lengths[i].data[0]])]

        return torch.cat(scores)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.start_idx]), tags])
        trans_score = Variable(torch.Tensor([0]))
        featu_score = Variable(torch.Tensor([0]))
        for i, feat in enumerate(feats):
            score += self.transit_matrix[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            trans_score += self.transit_matrix[tags[i + 1], tags[i]]
            featu_score += feat[tags[i + 1]]
        score = score + self.transit_matrix[self.stop_idx, tags[-1]]
        return score

    def v_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.label_num).fill_(-10000)
        vit[:, self.start_idx] = 0
        vit = Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transit_matrix.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transit_matrix[self.stop_idx].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)

        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths
