import json
import re
import torch
from torch._C import _disabled_torch_function_impl
from abc import abstractmethod
from typing import Callable

class Param(torch.Tensor):
    def __new__(cls, size, device=None, requires_grad=True, init: Callable = None):
        assert requires_grad, 'Parameters are expected to have gradients'

        data = torch.empty(size, device=device)
        if init is not None:
            init(data)
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

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self, named=True, prefix='', deep=True, buffers=False):
        for key, val in vars(self).items():
            if isinstance(val, Module) and deep:
                yield from val.parameters(named, prefix=f'{prefix + key}.', deep=deep, buffers=buffers)
            elif type(val) is Param:   # don't use isinstance, because Param is a subclass of Tensor
                if val.requires_grad:  # only trainable parameters
                    yield (prefix + key, val) if named else val
            elif buffers and key == '_buffers':
                for name, val in val.items():
                    yield (prefix + name, val) if named else val

    def modules(self, named=True, prefix=''):
        for key, val in vars(self).items():
            if isinstance(val, Module):
                yield (prefix + key, val) if named else val
                yield from val.modules(named, prefix=f'\t{prefix + key}.')

    def to(self, device):
        # print('module:', self)
        for name, param in self.parameters(deep=False, buffers=True): # todo: test the loop against parameters(deep=True)
            param.data = param.data.to(device)  # careful here
            # print(' -> param', name, param.device)
            if param.grad is not None:
                raise NotImplementedError
        for name, module in self.modules():
            module.to(device)
        return self

    def device_of_first_parameter(self):  # caution: a module can contain parameters on different devices
        return next(iter(self.parameters(named=False))).device

    def summary(self, params=False):
        print(self)
        for i, (name, module) in enumerate(self.modules()):
            prefix = re.match(r"\s*", name).group()
            print(f'\t{prefix}{i+1:3}.', module)
            if params:
                for param_name, param in module.parameters(deep=False):
                    print(f'\t\t{name}.{param_name}', list(param.size()))


    def register_buffer(self, name, value):
        self._buffers = getattr(self, '_buffers', {})
        self._buffers[name] = value
        return value

    def state_dict(self):
        return {k: v for k, v in self.parameters(buffers=True)}

    def load_state_dict(self, state_dict):
        for name, param in self.parameters(buffers=True):
            param.data[:] = state_dict[name]  # copy data, while validating its shape


    @property
    def n_params(self) -> str:
        num = self.count_params()
        return "{:,}".format(num)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters(named=False))

    def grad_norm(self):
        return torch.cat([param.grad.view(-1) for param in self.parameters(named=False)]).norm().item()

    def weight_norm(self):
        return torch.cat([p.view(-1) for name, p in self.parameters() if 'bias' not in name]).norm().item()

    def bias_norm(self):
        return torch.cat([p.view(-1) for name, p in self.parameters() if 'bias' in name]).norm().item()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self):
        return f'{self.__class__.__name__}(): {self.n_params} parameters'.replace(': 0 parameters', '')

class ModuleList(list, Module):
    def __init__(self, modules):
        super().__init__()
        for module in modules:
            if module is not None:  # sometimes None element is passed when the module is under condition (e.g. [... , Dropout(dropout_rate) if dropout_rate else None, ...]
                self.add(module)

    def add(self, module):
        assert isinstance(module, Module) or callable(module), f'Expected only Module instances or functions, but got: {module}'
        self.append(module)
        setattr(self, f'm{len(self)}', module)  # this is to make the modules discoverable (refer to self.modules())

    def __repr__(self):
        return f'{self.name}({len(self)}): {self.n_params} parameters'


class Sequential(ModuleList):

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, x, verbose=False):
        if verbose:
            print(list(x.shape), '<-', 'Input')
        for i, module in enumerate(self):
            try:
                if isinstance(module, Sequential):
                    x = module.forward(x, verbose=verbose)
                elif isinstance(module, Module):
                    x = module.forward(x)
                elif callable(module):
                    x = module(x)
                else:
                    raise Exception('Unexpected module: ' + type(module))
            except Exception as e:  # simplifies debugging
                print('ERROR', f'<- {i}.', module)
                raise e
            if verbose:
                print(list(x.shape), f'<- {i}.', module)
        # for name, module in self.modules():
        #     x = module.forward(x)
        return x
