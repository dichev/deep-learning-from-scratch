import torch
from math import sqrt

def xavier_normal_(tensor, in_size, out_size):
    std = sqrt(2. / (in_size + out_size))
    tensor.normal_(0, std)
    return tensor

def xavier_uniform_(tensor, in_size, out_size):
    a = -sqrt(6. / (in_size + out_size))
    b =  sqrt(6. / (in_size + out_size))
    tensor.uniform_(a, b)
    return tensor

def kaiming_normal_relu_(tensor, in_size, out_size=None):
    std = sqrt(2. / in_size)  # 2 is the ReLU's mean
    tensor.normal_(0, std)
    return tensor
