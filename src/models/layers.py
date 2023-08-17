import json
import torch
from functions import init
from models.parameters import Param

class Module:

    def parameters(self, named=True, prefix=''):
        for key, val in vars(self).items():
            if isinstance(val, Module):
                yield from val.parameters(named, prefix=f'{key}.')
            elif hasattr(val, '_is_parameter') and getattr(val, '_is_parameter') is True:
                yield (prefix + key, val) if named else val

    def summary(self):
        print(self)
        for key, val in self.parameters():
            print(f'\t{key}: {val.__class__.__name__}{list(val.size())}')

    def export(self, filename='./logs/model.json'):
        print('Export model to:', filename)
        params = {key: value.tolist() for key, value in self.parameters(named=True)}
        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)

    def __repr__(self):
        return f'{self.__class__.__name__}()'



class Linear(Module):
    def __init__(self, input_size, output_size=1, device='cpu', weights_init=init.kaiming_normal_relu):
        self.weight = Param(input_size, output_size, init=weights_init, device=device, requires_grad=True)  # (D, C)
        self.bias = Param(1, output_size, init=init.zeros, device=device, requires_grad=True)  # (D, C)
        self.input_size, self.output_size = input_size, output_size

    def forward(self, X):
        z = X @ self.weight + self.bias    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z

    def __repr__(self):
        return f'Linear({self.input_size}, {self.output_size})'


class Embedding(Module):  # aka lookup table
    def __init__(self, vocab_size, output_size, padding_idx=None, device='cpu'):
        self.weight = Param(vocab_size, output_size, init=init.normal, device=device, requires_grad=True)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx] = 0.

        self.input_size, self.output_size = vocab_size, output_size

    def forward(self, indices):
        assert torch.is_tensor(indices) and not torch.is_floating_point(indices), 'Use only tensor integer as indices, to avoid fancy indexing surprises'
        z = self.weight[indices]
        return z

    def __repr__(self):
        return f'Embedding({self.input_size}, {self.output_size})'

