import torch

def sign(x):
    return torch.where(x >= 0, 1, -1)

def unit_step(x):
    return torch.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def softmax(z):
    e = torch.exp(z - z.max())
    return e / e.sum(dim=-1, keepdim=True)


def log_softmax(z):
    e = z.exp().sum(dim=-1, keepdim=True)
    return z - e.log()  # that can be approximated as: z - max(z)


