import torch

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




