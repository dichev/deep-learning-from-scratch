import torch
from torch._C import _disabled_torch_function_impl

class Param(torch.Tensor):
    def __new__(cls, in_size, out_size, init, device=None, requires_grad=True):
        assert requires_grad, 'Parameters are expected to have gradients'

        data = init(in_size, out_size, device=device)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        return instance

    def __repr__(self, **kwargs):
        return torch.Tensor.__repr__(self, **kwargs).replace('Param', 'tensor')

    def __str__(self):
        return torch.Tensor.__str__(self).replace('Param', 'tensor')

    # __torch_function__ implementation wraps subclasses such that methods called on subclasses return a subclass instance instead of a torch.Tensor instance.
    # However, this is not desired here, since we want to return a torch.Tensor instance when doing math operations on Param instances.
    __torch_function__ = _disabled_torch_function_impl

