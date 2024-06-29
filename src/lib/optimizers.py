import torch
import math
import re


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

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_parameters')}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)


class SGD(Optimizer):

    def __init__(self, parameters, lr, weight_decay=0.):
        self.weight_decay = weight_decay
        super().__init__(parameters, lr)

    @torch.no_grad()
    def step(self):
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            param -= self.lr * self.weight_decay * param
            param -= self.lr * param.grad

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
            self.exp_avg[name] = beta * self.exp_avg[name] + param.grad
            param -= self.lr * self.exp_avg[name]

        return self


class AdaGrad(Optimizer):
    """
    + Adapts learning rate for each parameter - by scaling with the inverse of the sum of squares of past derivatives.
    - Diminishing learning rates - the sum of squares of past derivatives increases.
    """

    def __init__(self, parameters, lr, eps=1e-8):
        super().__init__(parameters, lr)
        self.eps = eps
        self.grad_sq = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.grad_sq[name] += param.grad ** 2
            adjusted_lr = self.lr / (self.grad_sq[name].sqrt() + self.eps)
            param -= adjusted_lr * param.grad

        return self


class RMSProp(Optimizer):
    """
    + Adapts learning rate for each parameter - by scaling with the exponential moving average of past derivatives.
    + No diminishing learning rates - because the exponential moving average
    - Initial bias - The initial running estimate is zero, causing a bias in the early iterations.
    """
    def __init__(self, parameters, lr, decay=0.9, eps=1e-8):
        super().__init__(parameters, lr)
        self.eps = eps
        self.decay = decay
        self.exp_avg_sq = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        beta = self.decay
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.exp_avg_sq[name] = beta * self.exp_avg_sq[name] + (1 - beta) * (param.grad ** 2)
            adjusted_lr = self.lr / (self.exp_avg_sq[name].sqrt() + self.eps)
            param -= adjusted_lr * param.grad

        return self


class AdaDelta(Optimizer):
    """
    + Adapts learning rate for each parameter - by scaling with the exponential moving average of past derivatives.
    + No diminishing learning rates - because the exponential moving average
    + No learning rate parameter
    """
    def __init__(self, parameters, lr=None, decay=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        self.eps = eps
        self.decay = decay
        self.exp_avg_sq = {name: torch.zeros_like(param) for name, param in self._parameters}
        self.exp_avg_sq_delta = {name: torch.zeros_like(param) for name, param in self._parameters}

    @torch.no_grad()
    def step(self):
        beta = self.decay
        for name, param in self._parameters:
            if param.grad is None:
                raise RuntimeError(f"Parameter {name} has no gradient")
            self.exp_avg_sq[name] = beta*self.exp_avg_sq[name] + (1-beta)*(param.grad**2)
            adjusted_lr = torch.sqrt((self.exp_avg_sq_delta[name] + self.eps) / (self.exp_avg_sq[name] + self.eps))
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
    def __init__(self, parameters, lr, eps=1e-8, momentum=0.9, decay=0.999):
        super().__init__(parameters, lr)
        self.eps = eps
        self.momentum = momentum
        self.decay = decay
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
            self.exp_avg[name] = beta1 * self.exp_avg[name] + (1 - beta1) * param.grad
            self.exp_avg_sq[name] = beta2 * self.exp_avg_sq[name] + (1 - beta2) * (param.grad**2)

            # bias_corrections
            exp_avg_corrected = self.exp_avg[name] / bias_correction1
            exp_avg_sq_corrected = self.exp_avg_sq[name] / bias_correction2

            param -= self.lr * exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + self.eps)

        return self


class AdamW(Adam):
    """
    Paper: Decoupled Weight Decay Regularization
    https://arxiv.org/pdf/1711.05101

    + Adapts learning rate for each parameter - by scaling with the exponential moving average of past derivatives.
    + No diminishing learning rates - because the exponential moving average
    + Initial bias correction
    + Reduces zigzagging (momentum) - by averaging previous gradients
    + Decoupled (from the loss optimization) weight decay (that's better than Adams' L2 regularization, where the decay affects the adaptive learning rates) (see https://arxiv.org/abs/1706.03762)
    """
    def __init__(self, parameters, lr, weight_decay=0, eps=1e-8, momentum=0.9, decay=0.999, weight_decay_filter=r''):
        super().__init__(parameters, lr, eps, momentum, decay)
        self.weight_decay = weight_decay
        self._parameters_decayed = []
        if weight_decay:
            self._parameters_decayed = [(name, param) for name, param in self._parameters if (not weight_decay_filter or re.search(weight_decay_filter, name))]

    @torch.no_grad()
    def step(self):
        # First decay parameters independently (same as regularization)
        for name, param in self._parameters_decayed:
            param -= self.lr * self.weight_decay * param

        # Then update parameters with the adapted learning rates
        super().step()
        return self


class LR_Scheduler:

    def __init__(self, optimizer, decay=.99, min_lr=1e-5):
        self.optimizer = optimizer
        self.steps = 0
        self.decay = decay
        self.min_lr = min_lr
        self.max_lr = optimizer.lr

    def step(self):  # must be called after each epoch
        self.steps += 1
        self.optimizer.lr = self.get_learn_rate(self.steps)

    def get_learn_rate(self, step):  # using discrete (compounded) decay instead exp(-self.decay), because the epochs are not continuous
        lr = self.max_lr * self.decay ** step
        return max(self.min_lr, lr)

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if not k == 'optimizer'}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)


class LR_StepScheduler(LR_Scheduler):

    def __init__(self, optimizer, step_size=10, decay=.99, min_lr=1e-5):
        super().__init__(optimizer, decay=decay, min_lr=min_lr)
        self.step_size = step_size

    def get_learn_rate(self, step):  # using discrete (compounded) decay instead exp(-self.decay), because the epochs are not continuous
        lr = self.max_lr * self.decay ** (step // self.step_size)
        return max(self.min_lr, lr)


class LR_PlateauScheduler(LR_Scheduler):

    def __init__(self, optimizer, patience=5, decay=.90, min_lr=1e-5, threshold=1e-5):
        super().__init__(optimizer, decay=decay, min_lr=min_lr)
        self.patience = patience
        self.steps = 0
        self.best = math.inf
        self.checkpoint = 0
        self.updated_steps = 0
        self.threshold = threshold

    def step(self, loss):
        self.steps += 1
        if loss < self.best - self.threshold:
            self.best = loss
            self.checkpoint = self.steps
        elif self.steps - self.checkpoint > self.patience:
            self.updated_steps += 1
            self.optimizer.lr = self.get_learn_rate(self.updated_steps)


class LR_CosineDecayScheduler:

    def __init__(self, optimizer, warmup_steps=20, decay_steps=200, min_lr=1e-5):
        assert warmup_steps >= 0 and decay_steps > 0
        self.optimizer = optimizer
        self.steps = 0
        self.min_lr = min_lr
        self.max_lr = self.optimizer.lr  # that's also the warmup target
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        if warmup_steps > 0:  # importantly set the initial lr to min_lr (in case the scheduler was called after optimizer.step)
            self.optimizer.lr = min_lr

    def step(self):  # must be called after each epoch
        self.steps += 1
        self.optimizer.lr = self.get_learn_rate(self.steps)

    def get_learn_rate(self, step):
        if step <= self.warmup_steps:
            return self.max_lr * step / self.warmup_steps

        if step > self.warmup_steps + self.decay_steps:
            return self.min_lr

        decay_ratio = (step - self.warmup_steps) / self.decay_steps
        cosine_decay = (1 + math.cos(decay_ratio * math.pi)) / 2
        decayed_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        return decayed_lr

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if not k == 'optimizer'}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)