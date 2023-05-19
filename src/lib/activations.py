import torch

def sign(x):
    return torch.where(x >= 0, 1, -1)

def unit_step(x):
    return torch.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))