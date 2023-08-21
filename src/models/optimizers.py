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

    def __init__(self, optimizer, decay=.99, min_lr=1e-5):
        self.optimizer = optimizer
        self.decay = decay
        self.min_lr = min_lr

    def step(self):  # must be called after each epoch, not after each batch
        optim = self.optimizer
        if optim.lr > self.min_lr:
            next_lr = optim.lr * self.decay  # using discrete (compounded) decay instead exp(-self.decay), because the epochs are not continuous
            optim.lr = max(self.min_lr, next_lr)


class LR_StepScheduler(LR_Scheduler):

    def __init__(self, optimizer, step_size=10, decay=.99, min_lr=1e-5):
        super().__init__(optimizer, decay=decay, min_lr=min_lr)
        self.step_size = step_size
        self.epoch = 0

    def step(self):  # must be called after each epoch, not after each batch
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            super().step()
