import torch
from math import sqrt


def init_zeros(in_size, out_size, device=None):
    return torch.zeros(in_size, out_size, device=device)

def init_normal(in_size, out_size, mu=0, std=1., device=None):
    tensor = torch.randn(in_size, out_size, device=device)
    tensor *= std
    tensor += mu
    return tensor

def init_xavier(in_size, out_size, device=None):
    tensor = torch.randn(in_size, out_size, device=device)
    tensor *= sqrt(2. / (in_size + out_size))
    return tensor


class Param(torch.Tensor):
    def __new__(cls, in_size, out_size, init, device=None, requires_grad=True):
        assert requires_grad, 'Parameters are expected to have gradients'

        data = init(in_size, out_size, device=device)
        instance = torch.Tensor._make_subclass(cls, data)
        instance._is_parameter = True  # will be used to detect all module's Parameters
        instance.requires_grad = True
        return instance

    def __repr__(self, **kwargs):
        return torch.Tensor.__repr__(self, **kwargs).replace('Param', 'tensor')

    def __str__(self):
        return torch.Tensor.__str__(self).replace('Param', 'tensor')


print(Param(3, 2, init=init_xavier, requires_grad=True, device='cuda'))

