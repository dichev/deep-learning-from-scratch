import torch

class Optimizer:

    def __init__(self, parameters, lr):
        assert callable(parameters), 'Expected the argument "parameter" to be an iterator function'
        self._parameters = parameters
        self.lr = lr

    @torch.no_grad()
    def step(self):
        for name, param in self._parameters():
            param -= self.lr * param.grad
            
        return self

    def zero_grad(self):
        for name, param in self._parameters():
            param.grad.zero_()


def nested(t):
    for i in t:
        if isinstance(i, tuple):
            yield from nested(i)
        else:
            yield i

# tup = ((1, 2, 3), (4, 5, (6, 7)), 8, 9)
# for elem in nested(tup):
#     print(elem)
