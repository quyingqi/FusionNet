#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2017/11/24
import torch
import torch.nn as nn
from .Constants import PAD


class SequenzeMLELoss(nn.Module):
    """
    Sequenze Maximum Likelihood Estimation Loss, with pad_index weight is 0.
    """
    def __init__(self, label_size, size_average=False, batch_average=False):
        super(SequenzeMLELoss, self).__init__()
        self.label_size = label_size
        self.size_average = size_average
        self.batch_average = batch_average
        if size_average and batch_average:
            raise RuntimeError("size_average, batch_average cannot both True")

        self.pad_index = PAD

        weight = torch.ones(label_size)
        weight[PAD] = 0
        self.loss = nn.CrossEntropyLoss(weight, size_average=False)

    def _check_size(self, pred, golden):
        batch_size, max_len, label_size = pred.size()
        assert batch_size == golden.size(0)
        assert max_len == golden.size(1)
        assert label_size == self.label_size

    def forward(self, pred, golden):
        """
        :param pred:   (batch, max_len, label_size)
        :param golden: (batch, max_len)
        :return:
        """
        loss = self.loss.forward(pred.view(-1, pred.size(-1)), golden.view(-1))
        if self.batch_average:
            # Average according to Batch
            batch_size = pred.size(0)
            return torch.div(loss, batch_size)
        elif self.size_average:
            # Average according to No-pad
            nopad_size = torch.sum(golden != self.pad_index).float()
            return torch.div(loss, nopad_size)
        return loss
