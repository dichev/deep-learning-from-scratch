import torch
from models.parameters import Param, init_xavier, init_zeros, init_normal


class Module:
    def parameters(self, named=True):
        for key, val in vars(self).items():
            if isinstance(val, Module):
                yield from val.parameters(named)
            elif hasattr(val, '_is_parameter') and getattr(val, '_is_parameter') is True:
                yield (key, val) if named else val


class Linear(Module):
    def __init__(self, input_size, output_size=1, device='cpu'):
        self.W = Param(input_size, output_size, init=init_xavier, device=device, requires_grad=True)  # (D, C)
        self.b = Param(1, output_size, init=init_zeros, device=device, requires_grad=True)  # (D, C)

    def forward(self, X):
        z = X @ self.W + self.b    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z


class Embedding(Module):  # aka lookup table
    def __init__(self, vocab_size, output_size, padding_idx=None, device='cpu'):
        self.W = Param(vocab_size, output_size, init=init_normal, device=device, requires_grad=True)
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

