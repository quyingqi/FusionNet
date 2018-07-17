#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/29
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from .base import get_rnn_cell
from ..mask_util import lengths2mask


class MaskBasedSingleDirectionRecurrentLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 dropout=0.3,
                 bias=True,
                 rnn_type="LSTM",
                 reverse=False):
        super(MaskBasedSingleDirectionRecurrentLayer, self).__init__()
        self.input_size = input_size
        self.cell_size = hidden_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        self.reverse = reverse
        self.bias = bias

        rnn_cell = get_rnn_cell(self.rnn_type)

        self.rnn_cell = rnn_cell(input_size=self.input_size,
                                 hidden_size=self.cell_size,
                                 bias=bias)

        self.output_size = self.hidden_size
        self.init_model()

    def init_model(self):
        for weight in self.rnn_cell.parameters():
            if len(weight.data.size()) == 2:
                nn.init.orthogonal(weight)

    def get_h0_state(self, x):

        batch_size, max_len, input_size = x.size()
        state_shape = 1, batch_size, self.cell_size

        if self.rnn_type == 'lstm':
            if isinstance(x, Variable):
                h0 = c0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = c0 = Variable(x.new(*state_shape).zero_())
            return h0, c0
        else:
            if isinstance(x, Variable):
                h0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = Variable(x.new(*state_shape).zero_())
            return h0

    def forward_step(self, x, h_t_1):
        """
        :param x:  (batch, input)
        :param h_t_1: (batch, hidden)
        :return:
        """
        if self.dropout > 0:
            x = f.dropout(x, p=self.dropout, training=self.training)
        return self.rnn_cell(x, h_t_1)

    def _forward(self, x, h0_state, reverse=None):
        batch_size, max_len, input_size = x.size()

        hidden_state_list = list()
        h_t_state = h0_state

        if self.dropout > 0:
            x = f.dropout(x, p=self.dropout, training=self.training)

        if reverse is None:
            reverse = self.reverse

        for i in range(max_len):
            if reverse:
                step_index = max_len - 1 - i
            else:
                step_index = i
            x_step = x[:, step_index, :]

            h_t_state = self.rnn_cell(x_step, h_t_state)

            if self.rnn_type == 'lstm':
                h_t, c_t = h_t_state
            else:
                h_t = h_t_state
            hidden_state_list.append(h_t)

        if reverse:
            hidden_state_list.reverse()

        return torch.stack(hidden_state_list, 1), h_t_state

    def _forward_mask(self, x, h0_state, mask, reverse=None):
        """
        :param x:        (batch, len, input_size)
        :param h0_state: (batch, hidden_size) / tuple(h0, c0)
        :param mask:     (batch, len)
        :param reverse:  reverse scan
        :return: hidden: (batch, len, hidden_size)
        """
        batch_size, max_len, input_size = x.size()

        hidden_state_list = list()
        h_t_1_state = h0_state

        if reverse is None:
            reverse = self.reverse

        for i in range(max_len):

            if reverse:
                step_index = max_len - 1 - i
            else:
                step_index = i
            x_step = x[:, step_index, :]
            m_step = mask[:, step_index].unsqueeze(-1).expand(batch_size, self.cell_size)
            h_t_state = self.rnn_cell(x_step, h_t_1_state)

            if self.rnn_type == 'lstm':
                h_t, c_t = h_t_state
                h_t_1, c_t_1 = h_t_1_state
                h_t = h_t * m_step + h_t_1 * (1. - m_step)
                c_t = c_t * m_step + c_t_1 * (1. - m_step)
                h_t_state = (h_t, c_t)
            else:
                h_t = h_t_state
                h_t_1 = h_t_1_state
                h_t = h_t * m_step + h_t_1 * (1. - m_step)
                h_t_state = h_t

            hidden_state_list.append(h_t)
            h_t_1_state = h_t_state

        if reverse:
            hidden_state_list.reverse()

        return torch.stack(hidden_state_list, 1), h_t_1_state

    def forward(self, x, h0_state=None, lengths=None, reverse=None):
        """
        :param x:        (batch, len, input_size)
        :param h0_state: (1, batch, hidden_size) / tuple(h0, c0)
        :param lengths:  (batch, )
        :param reverse:  reverse scan
        :return: hidden: (batch, len, hidden_size)
        """
        batch_size, max_len, input_size = x.size()

        if h0_state is None:
            h0_state = self.get_h0_state(x)

        # (1, batch, hidden_size) -> (batch, hidden_size)
        if self.rnn_type == 'lstm':
            h0_state = h0_state[0].squeeze(0), h0_state[1].squeeze(0)
        else:
            h0_state = h0_state[0].squeeze(0)

        if lengths is not None:
            mask = lengths2mask(lengths, max_len)
        else:
            mask = None

        if mask is None:
            output, hidden_state = self._forward(x, h0_state, reverse)
        else:
            output, hidden_state = self._forward_mask(x, h0_state, mask, reverse)

        if mask is not None:
            output = mask[:, :, None] * output

        # Add Dim (direction)
        if self.rnn_type == 'lstm':
            hidden_state = hidden_state[0].unsqueeze(0), hidden_state[1].unsqueeze(0)
        else:
            hidden_state = hidden_state.unsqueeze(0)

        return output, hidden_state


class MaskBasedBiDirectionRecurrentLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 dropout=0.3,
                 bias=True,
                 rnn_type="LSTM",
                 share_param=False):
        super(MaskBasedBiDirectionRecurrentLayer, self).__init__()
        self.input_size = input_size
        self.cell_size = hidden_size // 2
        self.hidden_size = self.cell_size * 2
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        self.bias = bias
        self.share_param = share_param

        if share_param:
            self.rnn = MaskBasedSingleDirectionRecurrentLayer(input_size,
                                                              hidden_size=self.cell_size,
                                                              dropout=self.dropout,
                                                              bias=self.bias,
                                                              rnn_type=self.rnn_type)
            self.rnn_reverse = None
        else:
            self.rnn = MaskBasedSingleDirectionRecurrentLayer(input_size,
                                                              hidden_size=self.cell_size,
                                                              dropout=self.dropout,
                                                              bias=self.bias,
                                                              rnn_type=self.rnn_type)
            self.rnn_reverse = MaskBasedSingleDirectionRecurrentLayer(input_size,
                                                                      hidden_size=self.cell_size,
                                                                      dropout=self.dropout,
                                                                      bias=self.bias,
                                                                      rnn_type=self.rnn_type,
                                                                      reverse=True)
        self.output_size = self.hidden_size
        self.init_model()

    def init_model(self):
        pass

    def get_h0_state(self, x):
        batch_size, max_len, input_size = x.size()
        state_shape = 2, batch_size, self.cell_size

        if self.rnn_type == 'lstm':
            if isinstance(x, Variable):
                h0 = c0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = c0 = Variable(x.new(*state_shape).zero_())
            return h0, c0
        else:
            if isinstance(x, Variable):
                h0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = Variable(x.new(*state_shape).zero_())
            return h0

    def forward(self, x, h0_state=None, lengths=None):
        """
        :param x:        (batch, len, input_size)
        :param h0_state: (2, batch, cell_size) / tuple
        :param lengths:  (batch)
        :return:
        """
        if h0_state is None:
            h0_state = self.get_h0_state(x)

        # keep (1, batch, cell_size)
        if self.rnn_type == 'lstm':
            h0_state_f = h0_state[0][:1], h0_state[1][:1]
            h0_state_b = h0_state[0][1:], h0_state[1][1:]
        else:
            h0_state_f = h0_state[:1]
            h0_state_b = h0_state[1:]

        if self.share_param:
            f_output, f_hidden_state = self.rnn.forward(x, h0_state_f, lengths)
            b_output, b_hidden_state = self.rnn.forward(x, h0_state_b, lengths, reverse=True)
        else:
            f_output, f_hidden_state = self.rnn.forward(x, h0_state_f, lengths)
            b_output, b_hidden_state = self.rnn_reverse.forward(x, h0_state_b, lengths, reverse=True)

        if self.rnn_type == 'lstm':
            hidden_state_h = torch.cat([f_hidden_state[0], b_hidden_state[0]], 0)
            hidden_state_c = torch.cat([f_hidden_state[1], b_hidden_state[1]], 0)
            hidden_state = hidden_state_h, hidden_state_c
        else:
            hidden_state = torch.cat([f_hidden_state, b_hidden_state], 0)

        return torch.cat([f_output, b_output], -1), hidden_state


class MaskBasedRNNLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 num_layers=1,
                 dropout=0.2,
                 brnn=True,
                 skip_brnn=False,
                 rnn_type="LSTM",
                 multi_layer_hidden='last',
                 share_param=False,
                 bias=True):
        super(MaskBasedRNNLayer, self).__init__()
        self.input_size = input_size
        self.skip_brnn = skip_brnn
        self.bidirectional = False if self.skip_brnn else brnn
        self.cell_size = (hidden_size // 2) if self.bidirectional else hidden_size
        self.hidden_size = (self.cell_size * 2) if self.bidirectional else self.cell_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        self.multi_layer_hidden = multi_layer_hidden
        self.share_param = share_param
        self.bias = bias

        self.rnn_layers = nn.ModuleList()
        input_dim = self.input_size
        for layer_i in range(num_layers):
            if self.bidirectional:
                rnn_layer = MaskBasedBiDirectionRecurrentLayer(input_size=input_dim,
                                                               hidden_size=self.hidden_size,
                                                               dropout=self.dropout,
                                                               bias=self.bias,
                                                               rnn_type=self.rnn_type,
                                                               share_param=self.share_param)
                input_dim = rnn_layer.output_size
                self.rnn_layers.append(rnn_layer)
            else:
                if self.skip_brnn:
                    is_reverse = layer_i % 2 == 1
                else:
                    is_reverse = False
                rnn_layer = MaskBasedSingleDirectionRecurrentLayer(input_size=input_dim,
                                                                   hidden_size=self.hidden_size,
                                                                   dropout=self.dropout,
                                                                   bias=self.bias,
                                                                   rnn_type=self.rnn_type,
                                                                   reverse=is_reverse)
                input_dim = rnn_layer.output_size
                self.rnn_layers.append(rnn_layer)

        if self.multi_layer_hidden == 'concatenate':
            self.output_size = self.hidden_size * num_layers
        else:
            self.output_size = self.hidden_size

        self.init_model()

    def init_model(self):
        pass

    def forward_step(self, x, h_state_t_1):
        """
        :param x:           (batch, input_size)
        :param h_state_t_1: (layer_num, batch, hidden_size)
        :return:
            output: multi_layer_hidden 'last' -> (batch, hidden_size)
                                       'concatenate' -> (batch, hidden_size * num_layer)
            h_t:    rnn_type 'GRU' -> (layer, batch, hidden_size)
                             'LSTM' -> tuple
        """
        if self.bidirectional or self.skip_brnn:
            raise RuntimeError("forward_step only for Single Direction")

        if self.rnn_type == "lstm":
            h_t_1, c_t_1 = h_state_t_1
            h_t, c_t = [], []
            x_i = x
            for layer_i in range(self.num_layers):
                h_i, c_i = self.rnn_layers[layer_i].forward_step(x_i, (h_t_1[layer_i], c_t_1[layer_i]))
                h_t.append(h_i)
                c_t.append(c_i)
                x_i = h_i
            # (layer) * (batch, hidden) -> (layer, batch, hidden)
            if self.multi_layer_hidden == 'last':
                output = h_t[-1]
            elif self.multi_layer_hidden == 'concatenate':
                output = torch.cat(h_t, dim=1)
            else:
                raise NotImplementedError
            h_t = torch.stack(h_t)
            c_t = torch.stack(c_t)
            return output, (h_t, c_t)

        elif self.rnn_type == 'gru':
            h_t_1 = h_state_t_1
            h_t = []
            x_i = x
            for layer_i in range(self.num_layers):
                h_i = self.rnn_layers[layer_i].forward_step(x_i, h_t_1[layer_i])
                h_t.append(h_i)
                x_i = h_i
            # (layer) * (batch, hidden) -> (layer, batch, hidden)
            if self.multi_layer_hidden == 'last':
                output = h_t[-1]
            elif self.multi_layer_hidden == 'concatenate':
                output = torch.cat(h_t, dim=1)
            else:
                raise NotImplementedError
            h_t = torch.stack(h_t)
            return output, h_t
        else:
            raise NotImplementedError

    def get_h0_state(self, x):
        """
        :param x: (batch, max_len, input_size)
        :return:  (num_layer * num_direction, batch, cell_size)
        """
        batch_size, max_len, input_size = x.size()
        num_direction = 2 if self.bidirectional else 1
        state_shape = self.num_layers * num_direction, batch_size, self.cell_size

        if self.rnn_type == 'lstm':
            if isinstance(x, Variable):
                h0 = c0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = c0 = Variable(x.new(*state_shape).zero_())
            return h0, c0
        else:
            if isinstance(x, Variable):
                h0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = Variable(x.new(*state_shape).zero_())
            return h0

    def forward(self, x, h0_state=None, lengths=None):
        """
        :param x:        (batch, max_len, input_size)
        :param h0_state: (num_layer * directions, batch, cell_size) / tuple
        :param lengths:  (batch,)
        :return:
            concatenate:
                (batch, max_len, hidden_size * layer)
            last:
                (batch, max_len, hidden_size)
            (num_layer * num_direction, batch, cell_size)
        """
        hidden_list = list()
        state_list = list()

        if h0_state is None:
            h0_state = self.get_h0_state(x)

        num_direction = 2 if self.bidirectional else 1

        pre_layer_hidden = x
        for layer_i in range(self.num_layers):

            h0_start = layer_i * num_direction
            h0_end = (layer_i + 1) * num_direction

            if self.rnn_type == 'lstm':
                hi_state = h0_state[0][h0_start:h0_end], h0_state[1][h0_start:h0_end]
            else:
                hi_state = h0_state[h0_start:h0_end]

            # (batch, max_len, cell_size * num_direction), (directions, batch, hidden_size)
            hi_output, hi_hidden_state = self.rnn_layers[layer_i].forward(pre_layer_hidden,
                                                                          h0_state=hi_state, lengths=lengths)

            pre_layer_hidden = hi_output

            # layer (batch, max_len, cell_size * num_direction)
            hidden_list.append(hi_output)
            # layer (directions, batch, hidden_size)
            state_list.append(hi_hidden_state)

        if self.rnn_type == 'lstm':
            h_hidden_state, c_hidden_state = zip(*state_list)
            hidden_state = torch.cat(h_hidden_state, 0), torch.cat(c_hidden_state, 0)
        else:
            hidden_state = torch.cat(state_list, 0)

        if self.multi_layer_hidden == 'concatenate':
            hidden_list = torch.cat(hidden_list, -1)
        elif self.multi_layer_hidden == 'last':
            hidden_list = hidden_list[-1]
        else:
            raise NotImplementedError

        return hidden_list, hidden_state


class MaskBasedRNNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=168,
                 num_layers=1,
                 dropout=0.2,
                 brnn=True,
                 skip_brnn=False,
                 rnn_type="LSTM",
                 multi_layer_hidden='last',
                 share_param=False,
                 bias=True):
        super(MaskBasedRNNEncoder, self).__init__()
        self.input_size = input_size
        self.skip_brnn = skip_brnn
        self.bidirectional = False if self.skip_brnn else brnn
        self.cell_size = (hidden_size // 2) if self.bidirectional else hidden_size
        self.hidden_size = (self.cell_size * 2) if self.bidirectional else self.cell_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type.lower()
        self.multi_layer_hidden = multi_layer_hidden
        self.share_param = share_param
        self.bias = bias

        if self.multi_layer_hidden == 'concatenate':
            self.output_size = self.hidden_size * num_layers
        else:
            self.output_size = self.hidden_size

        self.rnn = MaskBasedRNNLayer(input_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     dropout=self.dropout,
                                     brnn=self.bidirectional,
                                     skip_brnn=self.skip_brnn,
                                     rnn_type=self.rnn_type,
                                     multi_layer_hidden=self.multi_layer_hidden,
                                     share_param=self.share_param,
                                     bias=self.bias)

        self.init_model()

    def init_model(self):
        pass

    def get_h0_state(self, x):
        batch_size, max_len, input_size = x.size()
        num_direction = 2 if self.bidirectional else 1
        state_shape = self.num_layers * num_direction, batch_size, self.cell_size

        if self.rnn_type == 'lstm':
            if isinstance(x, Variable):
                h0 = c0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = c0 = Variable(x.new(*state_shape).zero_())
            return h0, c0
        else:
            if isinstance(x, Variable):
                h0 = Variable(x.data.new(*state_shape).zero_())
            else:
                h0 = Variable(x.new(*state_shape).zero_())
            return h0

    def forward_step(self, x, hidden):
        """

        :param x: (batch, 1, input_size)
        :param hidden: (num_layers * num_directions, batch, cell_size)
                       lstm tuple
        :return:
            (batch, cell_size * num_directions)
            (num_directions * num_layers, batch, cell_size)
            LSTM tuple
        """
        # (batch, 1, cell_size * num_directions) -> (batch, cell_size * num_directions)
        x = x.squeeze(1)
        # output (batch, cell_size * num_directions)
        # h_t    (num_directions * num_layers, batch, cell_size)
        #        LSTM tuple
        output, h_t = self.rnn.forward_step(x, hidden)
        # (batch, cell_size * num_directions) ->  (batch, 1, cell_size * num_directions)
        output = output.squeeze(1)
        return output, h_t

    def forward(self, x, lengths=None, hidden=None):
        """
        :param x:  (batch, max_len, input_size)
        :param lengths: (batch, )
        :param hidden:  (layer_num * direction, batch_size, cell_size)
                        tuple for LSTM
        :return:
            outputs:     multi_layer_hidden 'concatenate' -> (batch, len, cell_size * num_directions * num_layer)
                         multi_layer_hidden 'last' -> (batch, len, num_directions * cell_size)
            last_hidden: multi_layer_hidden 'concatenate' -> (batch, cell_size * num_directions * num_layer)
                         multi_layer_hidden 'last' -> (batch, cell_size * num_directions)
        """
        batch_size, max_len, input_size = x.size()

        '''
        outputs
            concatenate:
                (batch, max_len, hidden_size * layer)
            last:
                (batch, max_len, hidden_size)
        hidden_state
            (num_direction * num_layer, batch, cell_size)
        '''
        outputs, hidden_state = self.rnn.forward(x, lengths=lengths, h0_state=hidden)

        if self.rnn_type == 'lstm':
            hidden_state = hidden_state[0]

        if self.multi_layer_hidden == 'concatenate':
            # (num_direction * num_layer, batch, cell_size)
            #   -> (batch, cell_size * num_direction * num_layer)
            hidden = hidden_state.transpose(0, 1).contiguous().view(batch_size, -1)
        elif self.multi_layer_hidden == 'last':
            if self.bidirectional:
                # (num_direction, batch, cell_size)
                #   -> (batch, cell_size * num_direction)
                hidden = hidden_state[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                # (1, batch, cell_size)
                #   -> (batch, cell_size * 1)
                hidden = hidden_state[-1]
        else:
            raise NotImplementedError

        return outputs, hidden
