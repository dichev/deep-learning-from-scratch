import torch

class SGD:

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


