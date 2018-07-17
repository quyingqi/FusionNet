#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2017/11/17
import torch
import torch.nn as nn
from torch.autograd import Variable

from .attention import get_attention
from .recurrent import MaskBasedRNNEncoder, PaddBasedRNNEncoder, RNNEncoder
from .utils import aeq


class RNNDecoder(nn.Module):
    """
    Decoder recurrent neural network.
    """

    def __init__(self,
                 input_size, label_size,
                 context_size=168,
                 hidden_size=168,
                 num_layers=1,
                 dropout=0.2,
                 rnn_type="LSTM",
                 attention_type='MLPWordSeqAttention',
                 multi_layer_hidden='last',
                 bias=True):
        super(RNNDecoder, self).__init__()
        self.label_size = label_size
        self.input_size = input_size
        self.context_size = context_size
        self.num_layers = num_layers
        # TODO Decoder only One Lyaer
        assert num_layers == 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.attention_type = attention_type.lower()
        self.multi_layer_hidden = multi_layer_hidden
        self.bias = bias

        if self.attention_type is not None:
            self.attention = get_attention(self.attention_type)(input_size=self.hidden_size, seq_size=self.context_size)
        else:
            self.attention = None

        self.rnn = PaddBasedRNNEncoder(input_size=self.input_size + self.attention.output_size,
                                       hidden_size=self.hidden_size,
                                       num_layers=self.num_layers,
                                       dropout=self.dropout,
                                       brnn=False,
                                       rnn_type=self.rnn_type,
                                       multi_layer_hidden=self.multi_layer_hidden,
                                       bias=self.bias)

        self.decoder = nn.Linear(in_features=self.rnn.output_size, out_features=self.label_size, bias=False)

    def share_decoder_weight(self, weight):
        """
        Share Decoder Weight with Word Embedding
        :param weight: Variable
        :return:
        """
        assert weight.size() == self.decoder.weight.size()
        assert isinstance(weight, Variable)
        self.decoder.weight = weight

    def init_decoder_weight(self, weight):
        """
        Init Decoder Weight
        :param weight: Tensor
        :return:
        """
        assert weight.size() == self.decoder.weight.size()
        assert isinstance(weight, torch.Tensor)
        self.decoder.weight.data = weight

    def get_h0_state(self, x):
        return self.rnn.get_h0_state(x)

    def _decode(self, x, lengths=None, hidden=None):
        """
        Args:
            x (LongTensor): batch x len x input_size
            lengths (LongTensor): batch
            hidden: Initial hidden state.
                    (num_layer, batch, hidden_size)

        Returns:
            decoded: (batch, max_len, dict_size)
        """
        batch_size, max_len, input_size = x.size()

        if hidden is None:
            h0_state = self.rnn.get_h0_state(x)
        else:
            h0_state = hidden

        if lengths is not None:
            _batch_size = lengths.size(0)
            aeq(batch_size, _batch_size)

        # (batch, max_len, hidden_size), (num_layer * num_direction, batch, cell_size)
        hidden_list, hidden_state = self.rnn.forward(x, hidden=h0_state, lengths=lengths)

        # (batch, max_len, hidden_size) -> # (batch * max_len, hidden_size)
        to_decode = hidden_list.view(batch_size * max_len, -1)

        # (batch * max_len, dict_size) -> (batch, max_len, dict_size)
        decoded = self.decoder.forward(to_decode).view(batch_size, max_len, -1)

        return decoded

    def _decode_with_context(self, x, lengths=None, hidden=None, context=None, context_lengths=None):
        """
        Args:
            x (LongTensor): batch x len x input_size
            lengths (LongTensor): batch
            hidden: Initial hidden state.
                    (num_layer, batch, hidden_size)
            context: Context Hidden State
                    (batch, src_len, hidden_size)

        Returns:
            decoded: (batch, max_len, dict_size)
        """
        if hidden is None:
            hidden = self.get_h0_state(x)

        batch_size, max_len, input_size = x.size()

        if hidden is None:
            h_state = self.rnn.get_h0_state(x)
        else:
            h_state = hidden

        if lengths is not None:
            _batch_size = lengths.size(0)
            aeq(batch_size, _batch_size)

        decoded_list = list()

        for _x in torch.split(x, split_size=1, dim=1):
            _x = _x.squeeze(1)
            # decode_t (batch, dict_label_size)
            # h_state  rnn_type 'GRU' -> (layer, batch, hidden_size)
            #                   'LSTM' -> tuple
            decode_t, h_state = self.decode_step_with_context(x=_x, hidden=h_state,
                                                              context=context, context_lengths=context_lengths)
            # [(batch, dict_label_size), (batch, dict_label_size), (batch, dict_label_size)]
            decoded_list.append(decode_t)

        # list[ (batch, dict_label_size) ]
        #   -> (batch, max_len, dict_label_size)
        return torch.stack(decoded_list, 1)

    def decode_step(self, x, hidden):
        """
        :param x:       (batch, input_size)
        :param hidden:  (batch, hidden_size) # GRU / RNN
                        tuple # LSTM
        :return:
            decoded: (batch, dict_label_size)
            h_t:    rnn_type 'GRU' -> (layer, batch, hidden_size)
                             'LSTM' -> tuple
        """
        '''
        output: multi_layer_hidden 'last' -> (batch, hidden_size)
                                   'concatenate' -> (batch, hidden_size * num_layer)
        h_t:    rnn_type 'GRU' -> (layer, batch, hidden_size)
                         'LSTM' -> tuple
        '''
        output, h_t = self.rnn.forward_step(x, hidden)

        # (batch, rnn_output_size) -> (batch, dict_label_size)
        decoded = self.decoder.forward(output)

        return decoded, h_t

    def decode_step_with_context(self, x, hidden, context, context_lengths=None):
        """
        :param x:       (batch, input_size)
        :param hidden:  (layer, batch, hidden_size) # GRU / RNN
                        tuple # LSTM
        :param context: (batch, src_len, context_size)
        :param context_lengths: (batch, )
        :return:
            decoded: (batch, dict_label_size)
            h_t:    rnn_type 'GRU' -> (layer, batch, hidden_size)
                             'LSTM' -> tuple
        """
        # TODO only for one layer
        assert self.num_layers == 1

        if self.rnn_type == 'lstm':
            h_t_1_state = hidden[0]
        else:
            h_t_1_state = hidden

        # (batch, context_size), (batch, src_len)
        attention_context, attention_weight = self.attention.forward(x=h_t_1_state[0], seq=context,
                                                                     lengths=context_lengths)

        # (batch, input_size) (batch, context_size) -> (batch, input_size + context_size)
        decode_rnn_input = torch.cat([x, attention_context], 1)
        # (batch, input_size + context_size) -> (batch, 1, input_size + context_size)
        decode_rnn_input = decode_rnn_input.unsqueeze(1)

        '''
        output: multi_layer_hidden 'last' -> (batch, hidden_size)
                                   'concatenate' -> (batch, hidden_size * num_layer)
        h_t:    rnn_type: 'GRU' -> (batch, hidden_size)
                          'LSTM' -> tuple
        '''
        output, h_t = self.rnn.forward_step(x=decode_rnn_input, hidden=hidden)

        # (batch, rnn_output_size) -> (batch, dict_label_size)
        decoded = self.decoder.forward(output)

        return decoded, h_t

    def forward(self, x, lengths=None, hidden=None, context=None, context_lengths=None):
        if context is None:
            return self._decode(x, lengths=lengths, hidden=hidden)
        else:
            assert self.attention is not None
            return self._decode_with_context(x, lengths=lengths, hidden=hidden,
                                             context=context, context_lengths=context_lengths)
