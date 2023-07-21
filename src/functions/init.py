import torch
from math import sqrt


def zeros(in_size, out_size, device=None):
    return torch.zeros(in_size, out_size, device=device)

def uniform(in_size, out_size, a=-1, b=1, device=None):
    return torch.rand(in_size, out_size, device=device)*(b-a) + a

def normal(in_size, out_size, mu=0, std=1., device=None):
    tensor = torch.randn(in_size, out_size, device=device)
    tensor *= std
    tensor += mu
    return tensor

def xavier_normal(in_size, out_size, device=None):
    tensor = torch.randn(in_size, out_size, device=device)
    tensor *= sqrt(2. / (in_size + out_size))
    return tensor

def xavier_uniform(in_size, out_size, device=None):
    a = -sqrt(6. / (in_size + out_size))
    b =  sqrt(6. / (in_size + out_size))
    tensor = torch.rand(in_size, out_size, device=device)*(b-a) + a
    return tensor

