# -*- coding:utf-8 -*- 
# Author: Roger
import torch
from torch.autograd import Variable


def lengths2mask(lengths, max_length, byte=False):
    batch_size = lengths.size(0)
    # print max_length
    # print torch.max(lengths)[0]
    # assert max_length == torch.max(lengths)[0]
    range_i = torch.arange(0, max_length).expand(batch_size, max_length).long()
    if lengths.is_cuda:
        range_i = range_i.cuda(lengths.get_device())
    if isinstance(lengths, Variable):
        range_i = Variable(range_i)
    lens = lengths.unsqueeze(-1).expand(batch_size, max_length)
    mask = lens > range_i
    if byte:
        return mask
    else:
        return mask.float()


def relative_postition2mask(start, end, max_length):
    """
    :param start: Start Position
    :param end:   End Position
    :param max_length: Max Length, so max length must big or equal than max of end
    :return:
    """
    if isinstance(end, Variable):
        # print torch.max(end).data[0], max_length
        assert torch.max(end).data[0] <= max_length
    return lengths2mask(end, max_length) - lengths2mask(start, max_length)


def test_split():
    zero = torch.zeros(7).int()
    left_position = torch.LongTensor([0, 3, 4, 5, 4, 7, 5])
    right_position = torch.LongTensor([3, 4, 6, 7, 8, 9, 8])
    lens = torch.LongTensor([5, 7, 8, 8, 10, 12, 9])
    left_position = Variable(left_position)
    right_position = Variable(right_position)
    zero = Variable(zero)
    lens = Variable(lens)
    left = relative_postition2mask(zero, left_position, torch.max(lens).data[0])
    middle = relative_postition2mask(left_position, right_position, torch.max(lens).data[0])
    right = relative_postition2mask(right_position, lens, torch.max(lens).data[0])
    print(left)
    print(middle)
    print(right)
    print(left + middle + right)


if __name__ == "__main__":
    test_split()
