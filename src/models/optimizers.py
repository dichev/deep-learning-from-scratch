import torch
import math


class Optimizer:

    def __init__(self, parameters, lr):
        assert hasattr(parameters, '__iter__'), f'Expected the argument "parameters" to be an iterable, but got {type(parameters)}'
        self._parameters = list(parameters)

        param_names = [name for name, param in self._parameters]
        assert len(param_names) == len(set(param_names)), f'Expected unique parameter names, but got {param_names}'

        self.lr = lr

    @torch.no_grad()
    def step(self):
        for name, param in self._parameters:
            param += -self.lr * param.grad

        return self

    def zero_grad(self):
        for name, param in self._parameters:
            param.grad.zero_()


class SGD(Optimizer):

    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)


class SGD_Momentum(Optimizer):

    def __init__(self, parameters, lr, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        beta = self.momentum
        for name, param in self._parameters:
            V = self.velocities[name]
            self.velocities[
                name] = beta * V - self.lr * param.grad  # *(1-beta) term is assumed to be integrated into the learning rate
            param += self.velocities[name]

        return self


class AdaGrad(Optimizer):

    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)
        self.eps = 1e-8
        self.magnitudes = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        for name, param in self._parameters:
            self.magnitudes[name] += param.grad ** 2
            lr_reduce = torch.sqrt(self.magnitudes[name] + self.eps)
            param += -(self.lr / lr_reduce) * param.grad

        return self


class LR_Scheduler:

    def __init__(self, optimizer, decay=.99, min_lr=1e-5):
        self.optimizer = optimizer
        self.decay = decay
        self.min_lr = min_lr

    def step(self):  # must be called after each epoch, not after each batch
        self.reduce_lr()

    def reduce_lr(self):
        optim = self.optimizer
        if optim.lr > self.min_lr:
            next_lr = optim.lr * self.decay  # using discrete (compounded) decay instead exp(-self.decay), because the epochs are not continuous
            optim.lr = max(self.min_lr, next_lr)


class LR_StepScheduler(LR_Scheduler):

    def __init__(self, optimizer, step_size=10, decay=.99, min_lr=1e-5):
        super().__init__(optimizer, decay=decay, min_lr=min_lr)
        self.step_size = step_size
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.reduce_lr()


class LR_PlateauScheduler(LR_Scheduler):

    def __init__(self, optimizer, patience=5, decay=.90, min_lr=1e-5, threshold=1e-5):
        super().__init__(optimizer, decay=decay, min_lr=min_lr)
        self.patience = patience
        self.epoch = 0
        self.best = math.inf
        self.threshold = threshold

    def step(self, loss):
        self.epoch += 1
        if loss < self.best - self.threshold:
            self.best = loss
            self.epoch = 0
        elif self.epoch > self.patience:
            self.reduce_lr()
            self.epoch = 0
