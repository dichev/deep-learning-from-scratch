import torch
import math


class Optimizer:

    def __init__(self, parameters, lr):
        assert hasattr(parameters, '__iter__'), f'Expected the argument "parameters" to be an iterable, but got {type(parameters)}'
        self._parameters = list(parameters)
        assert len(self._parameters) > 0, 'No parameters!'

        param_names = [name for name, param in self._parameters]
        assert len(param_names) == len(set(param_names)), f'Expected unique parameter names, but got {param_names}'

        self.lr = lr

    @torch.no_grad()
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for name, param in self._parameters:
            if param.grad is not None:
                param.grad.zero_()


class SGD(Optimizer):

    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)

    @torch.no_grad()
    def step(self):
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            param += -self.lr * param.grad

        return self


class SGD_Momentum(Optimizer):
    """
    + Reduces zigzagging - by averaging previous gradients
    - Same learning rate for all parameters
    """

    def __init__(self, parameters, lr, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.exp_avg = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        beta = self.momentum
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.exp_avg[name] = beta * self.exp_avg[name] + (1 - beta) * param.grad
            param -= self.lr * self.exp_avg[name]

        return self


class AdaGrad(Optimizer):
    """
    + Adapts learning rate for each parameter - by scaling with the inverse of the sum of squares of past derivatives.
    - Diminishing learning rates - the sum of squares of past derivatives increases.
    """

    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)
        self.eps = 1e-8
        self.grad_sq = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.grad_sq[name] += param.grad ** 2
            adjusted_lr = self.lr / (self.grad_sq[name] + self.eps).sqrt()
            param -= adjusted_lr * param.grad

        return self


class RMSProp(Optimizer):
    """
    + Adapts learning rate for each parameter - by scaling with the exponential moving average of past derivatives.
    + No diminishing learning rates - because the exponential moving average
    - Initial bias - The initial running estimate is zero, causing a bias in the early iterations.
    """
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)
        self.eps = 1e-8
        self.decay = 0.9
        self.exp_avg_sq = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        beta = self.decay
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.exp_avg_sq[name] = beta * self.exp_avg_sq[name] + (1 - beta) * (param.grad ** 2)
            adjusted_lr = self.lr / (self.exp_avg_sq[name] + self.eps).sqrt()
            param -= adjusted_lr * param.grad

        return self


class AdaDelta(Optimizer):
    """
    + Adapts learning rate for each parameter - by scaling with the exponential moving average of past derivatives.
    + No diminishing learning rates - because the exponential moving average
    + No learning rate parameter
    """
    def __init__(self, parameters, lr=None):
        super().__init__(parameters, lr)
        self.eps = 1e-8
        self.decay = 0.99
        self.exp_avg_sq = {name: torch.zeros_like(param) for name, param in self._parameters}
        self.exp_avg_sq_delta = {name: torch.ones_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        beta = self.decay
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.exp_avg_sq[name] = beta*self.exp_avg_sq[name] + (1-beta)*(param.grad**2)
            adjusted_lr = torch.sqrt(self.exp_avg_sq_delta[name] / (self.exp_avg_sq[name] + self.eps))
            param_delta = adjusted_lr * param.grad
            self.exp_avg_sq_delta[name] = beta * self.exp_avg_sq_delta[name] + (1 - beta) * (param_delta ** 2)
            param -= param_delta

        return self

class Adam(Optimizer):
    """
    + Adapts learning rate for each parameter - by scaling with the exponential moving average of past derivatives.
    + No diminishing learning rates - because the exponential moving average
    + Initial bias correction
    + Reduces zigzagging (momentum) - by averaging previous gradients
    """
    def __init__(self, parameters, lr):
        super().__init__(parameters, lr)
        self.eps = 1e-8
        self.momentum = 0.9
        self.decay = 0.999
        self.exp_avg = {name: torch.zeros_like(param) for name, param in self._parameters}     # i.e. first moment
        self.exp_avg_sq = {name: torch.zeros_like(param) for name, param in self._parameters}  # i.e. second moment
        self.steps = 0

    @torch.no_grad()
    def step(self):
        self.steps += 1
        beta1 = self.momentum
        beta2 = self.decay
        t = self.steps
        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t

        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.exp_avg[name] = beta1 * self.exp_avg[name] + (1 - beta1) * self.lr * param.grad
            self.exp_avg_sq[name] = beta2 * self.exp_avg_sq[name] + (1 - beta2) * (param.grad**2)

            # bias_corrections
            exp_avg_corrected = self.exp_avg[name] / bias_correction1
            exp_avg_sq_corrected = self.exp_avg_sq[name] / bias_correction2

            adjusted_lr = self.lr / (exp_avg_sq_corrected + self.eps).sqrt()
            param -= adjusted_lr * exp_avg_corrected

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
