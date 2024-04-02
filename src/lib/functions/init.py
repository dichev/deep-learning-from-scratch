import torch
from math import sqrt


def xavier_normal_(tensor, in_size, out_size):
    std = sqrt(2. / (in_size + out_size))
    tensor.normal_(0, std)
    return tensor


def xavier_uniform_(tensor, in_size, out_size):
    bound = sqrt(6. / (in_size + out_size))
    tensor.uniform_(-bound, bound)
    return tensor


def kaiming_normal_relu_(tensor, in_size, out_size=None):
    gain = sqrt(2)  # 2 is the ReLU's mean
    std = gain * sqrt(1 / in_size)
    tensor.normal_(0, std)
    return tensor


def kaiming_uniform_relu_(tensor, in_size, out_size=None):
    gain = sqrt(2)  # 2 is the ReLU's mean
    bound = gain * sqrt(3 / in_size)
    tensor.uniform_(-bound, bound)
    return tensor


def linear_uniform_(tensor, in_size, out_size=None):  # That is pytorch's default initialization for Linear layers
    bound = 1 / sqrt(in_size)
    tensor.uniform_(-bound, bound)
    return tensor
