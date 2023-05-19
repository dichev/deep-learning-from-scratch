import torch

class Optimizer:

    def __init__(self, params, lr):
        self._params = params
        self.lr = lr

    @torch.no_grad()
    def step(self):
        for param in self._params:
            param -= self.lr * param.grad
            
        return self

    def zero_grad(self):
        for param in self._params:
            param.grad.zero_()