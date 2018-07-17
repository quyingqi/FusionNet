#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/11/29
import torch.nn as nn


def get_rnn(rnn_type):
    if rnn_type == "lstm":
        rnn = nn.LSTM
    elif rnn_type == "gru":
        rnn = nn.GRU
    elif rnn_type == "rnn":
        rnn = nn.RNN
    else:
        raise NotImplementedError("RNN Type: LSTM GRU RNN")
    return rnn


def get_rnn_cell(rnn_type):
    if rnn_type == "lstm":
        rnn_cell = nn.LSTMCell
    elif rnn_type == "gru":
        rnn_cell = nn.GRUCell
    elif rnn_type == "rnn":
        rnn_cell = nn.RNNCell
    else:
        raise NotImplementedError("RNN Type: LSTM GRU RNN")
    return rnn_cell
