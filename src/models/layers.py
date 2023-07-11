import torch


class Linear:
    def __init__(self, input_size, output_size=1, device='cpu'):
        self.W = torch.randn(input_size, output_size, device=device, requires_grad=True)  # (D, C)
        self.b = torch.randn(1, output_size, device=device, requires_grad=True)           # (1, C)
        self.params = (self.W, self.b)  # todo: consider extending torch.nn.Module

    def forward(self, X):
        z = X @ self.W + self.b    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z


class Embedding:  # aka lookup table
    def __init__(self, vocab_size, output_size, padding_idx=None, device='cpu'):
        self.W = torch.randn(vocab_size, output_size, device=device)
        if padding_idx is not None:
            with torch.no_grad():
                self.W[padding_idx] = 0.

        self.input_size, self.output_size = vocab_size, output_size
        self.params = (self.W, )

    def forward(self, indices):
        assert isinstance(indices, torch.LongTensor), 'Use only LongTensor as indices, to avoid fancy indexing surprises'
        z = self.W[indices]
        return z

    def __str__(self):
        return f'Embedding({self.input_size}, {self.output_size})'

