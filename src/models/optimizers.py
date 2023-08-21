import torch
import math

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


class LR_Scheduler:

    def __init__(self, optimizer, exp_decay=0, min_lr=1e-5):
        self.optimizer = optimizer
        self.exp_decay = exp_decay
        self.min_lr = min_lr

    def step(self):  # must be called after each epoch, not after each batch
        optim = self.optimizer
        if optim.lr > self.min_lr:
            next_lr = optim.lr * (1 - self.exp_decay)  # using discrete (compounded) decay instead exp(-self.exp_decay), because the epochs are not continuous
            optim.lr = max(self.min_lr, next_lr)

