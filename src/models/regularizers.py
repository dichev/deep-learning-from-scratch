import torch

def L2_regularizer(parameters, lambd=1.):
    norm = 0.
    for name, param in parameters:
        norm += lambd * (param**2).sum()
    return norm

def L1_regularizer(parameters, lambd=1.):
    norm = 0.
    for name, param in parameters:
        norm += lambd * param.abs().sum()
    return norm

def elastic_regularizer(parameters, lambd=1., alpha=.5):
    norm = alpha * L1_regularizer(parameters, lambd) + (1 - alpha) * L2_regularizer(parameters, lambd)
    return norm

def grad_clip_(params, limit_value):
    for name, param in params:
        param.grad.clamp_(-limit_value, limit_value)

def grad_clip_norm_(params, max_norm):
    grads = [param.grad for name, param in params]
    norm = torch.cat([grad.view(-1) for grad in grads]).norm().item()
    if norm > max_norm:
        for grad in grads:
            grad *= max_norm / norm

    return min(norm, max_norm)

