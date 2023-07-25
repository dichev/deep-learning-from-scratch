import torch
from functions import init
from models.parameters import Param


class Module:
    def parameters(self, named=True):
        for key, val in vars(self).items():
            if isinstance(val, Module):
                yield from val.parameters(named)
            elif hasattr(val, '_is_parameter') and getattr(val, '_is_parameter') is True:
                yield (key, val) if named else val


class Linear(Module):
    def __init__(self, input_size, output_size=1, device='cpu', weights_init=init.kaiming_normal_relu):
        self.W = Param(input_size, output_size, init=weights_init, device=device, requires_grad=True)  # (D, C)
        self.b = Param(1, output_size, init=init.zeros, device=device, requires_grad=True)  # (D, C)

    def forward(self, X):
        z = X @ self.W + self.b    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z


class Embedding(Module):  # aka lookup table
    def __init__(self, vocab_size, output_size, padding_idx=None, device='cpu'):
        self.W = Param(vocab_size, output_size, init=init.normal, device=device, requires_grad=True)
        if padding_idx is not None:
            with torch.no_grad():
                self.W[padding_idx] = 0.

        self.input_size, self.output_size = vocab_size, output_size

    def forward(self, indices):
        assert torch.is_tensor(indices) and not torch.is_floating_point(indices), 'Use only tensor integer as indices, to avoid fancy indexing surprises'
        z = self.W[indices]
        return z

    def __str__(self):
        return f'Embedding({self.input_size}, {self.output_size})'

