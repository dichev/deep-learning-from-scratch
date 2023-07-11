import torch


class Linear:
    def __init__(self, input_size, output_size=1, device='cpu'):
        self.W = torch.rand(input_size, output_size, device=device, requires_grad=True)  # (D, C)
        self.b = torch.rand(1, output_size, device=device, requires_grad=True)           # (1, C)
        self.params = (self.W, self.b) # todo: consider extending torch.nn.Module

    def forward(self, X):
        z = X @ self.W + self.b    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z
