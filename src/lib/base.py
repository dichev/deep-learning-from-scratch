import json
import re
import torch
from torch._C import _disabled_torch_function_impl

class Param(torch.Tensor):
    def __new__(cls, size, device=None, requires_grad=True):
        assert requires_grad, 'Parameters are expected to have gradients'

        data = torch.empty(size, device=device)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        return instance

    def __repr__(self, **kwargs):
        return torch.Tensor.__repr__(self, **kwargs).replace('Param', 'tensor')

    def __str__(self):
        return torch.Tensor.__str__(self).replace('Param', 'tensor')

    # __torch_function__ implementation wraps subclasses such that methods called on subclasses return a subclass instance instead of a torch.Tensor instance.
    # However, this is not desired here, since we want to return a torch.Tensor instance when doing math operations on Param instances.
    __torch_function__ = _disabled_torch_function_impl


class Module:

    def forward(self, x):
        raise NotImplementedError

    def parameters(self, named=True, prefix='', deep=True):
        for key, val in vars(self).items():
            if isinstance(val, Module) and deep:
                yield from val.parameters(named, prefix=f'{prefix + key}.')
            elif type(val) is Param:   # don't use isinstance, because Param is a subclass of Tensor
                if val.requires_grad:  # only trainable parameters
                    yield (prefix + key, val) if named else val

    def modules(self, named=True, prefix=''):
        for key, val in vars(self).items():
            if isinstance(val, Module):
                yield (prefix + key, val) if named else val
                yield from val.modules(named, prefix=f'\t{prefix + key}.')

    def summary(self, params=False):
        print(self)
        for i, (name, module) in enumerate(self.modules()):
            prefix = re.match(r"\s*", name).group()
            print(f'\t{prefix}{i+1:3}.', module)
            if params:
                for param_name, param in module.parameters(deep=False):
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

    @property
    def n_params(self):
        num = sum(p.numel() for p in self.parameters(named=False))
        return "{:,}".format(num)

    def grad_norm(self):
        return torch.cat([param.grad.view(-1) for param in self.parameters(named=False)]).norm().item()

    def weight_norm(self):
        return torch.cat([p.view(-1) for name, p in self.parameters() if 'bias' not in name]).norm().item()

    def bias_norm(self):
        return torch.cat([p.view(-1) for name, p in self.parameters() if 'bias' in name]).norm().item()

    def __repr__(self):
        return f'{self.__class__.__name__}(): {self.n_params} parameters'

