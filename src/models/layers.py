import json
import torch
from functions import init
from models.parameters import Param

class Module:

    def parameters(self, named=True, prefix=''):
        for key, val in vars(self).items():
            if isinstance(val, Module):
                yield from val.parameters(named, prefix=f'{key}.')
            elif type(val) is Param:  # don't use isinstance, because Param is a subclass of Tensor
                yield (prefix + key, val) if named else val

    def modules(self, named=True, prefix=''):
        for key, val in vars(self).items():
            if isinstance(val, Module):
                yield from val.modules(named, prefix=f'{key}.')
                yield (prefix + key, val) if named else val

    def summary(self):
        print(self)
        for name, module in self.modules():
            print(f'\t{name}:', module)
            for param_name, param in module.parameters():
                print(f'\t\t{name}.{param_name}', list(param.size()))

    def export(self, filename='./runs/model.json'):
        print('Export model to:', filename)

        network = {'layers': []}
        for name, module in self.modules():
            layer = {
                'type': module.__class__.__name__,
                'name': name,
            }
            for param_name, param in module.parameters():
                layer[param_name] = param.tolist()
            network['layers'].append(layer)

        with open(filename, 'w') as f:
            json.dump(network, f, indent=2)

    def __repr__(self):
        input = self.input_size if hasattr(self, 'input_size') else ''
        output = self.output_size if hasattr(self, 'output_size') else ''
        return f'{self.__class__.__name__}({input}, {output})'


class Linear(Module):
    def __init__(self, input_size, output_size=1, device='cpu', weights_init=init.kaiming_normal_relu):
        self.weight = Param(input_size, output_size, init=weights_init, device=device, requires_grad=True)  # (D, C)
        self.bias = Param(1, output_size, init=init.zeros, device=device, requires_grad=True)  # (D, C)
        self.input_size, self.output_size = input_size, output_size

    def forward(self, X):
        z = X @ self.weight + self.bias    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z

    def __repr__(self):
        return f'Linear({self.input_size}, {self.output_size}, bias=true)'


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
        return f'Embedding({self.input_size}, {self.output_size}, bias=false)'


class BatchNorm(Module):

    def __init__(self, size, device='cpu'):
        self.beta = Param(1, size, init=init.zeros, device=device, requires_grad=True)
        self.gamma = Param(1, size, init=init.ones, device=device, requires_grad=True)

        self.running_mean = torch.zeros(1, size, device=device)
        self.running_var = torch.ones(1, size, device=device)
        self.decay = 0.9

    def forward(self, x):
        # mini-batch statistics
        if torch.is_grad_enabled():
            assert len(x) > 1, 'BatchNorm layer requires at least 2 samples in batch'

            mu, var = x.mean(dim=0), x.var(dim=0)
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * mu
            self.running_var  = self.decay * self.running_var  + (1 - self.decay) * var
        else:
            mu, var = self.running_mean, self.running_var

        # normalize x along the mini-batch
        x = (x - mu) / (var + 1e-5).sqrt()
        x = self.gamma * x + self.beta

        return x

