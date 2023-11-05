import torch
from functions import init
from functions.activations import tanh
from models.base import Param, Module

class Linear(Module):
    def __init__(self, input_size, output_size=1, device='cpu', weights_init=init.kaiming_normal_relu):
        self.weight = Param(input_size, output_size, init=weights_init, device=device, requires_grad=True)  # (D, C)
        self.bias = Param(1, output_size, init=init.zeros, device=device, requires_grad=True)  # (D, C)
        self.input_size, self.output_size = input_size, output_size

    def forward(self, X):
        z = X @ self.weight + self.bias    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z

    def __repr__(self):
        return f'Linear({self.input_size}, {self.output_size}, bias=true)'


class Embedding(Module):  # aka lookup table
    def __init__(self, vocab_size, output_size, padding_idx=None, device='cpu'):
        self.weight = Param(vocab_size, output_size, init=init.normal, device=device, requires_grad=True)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx] = 0.

        self.input_size, self.output_size = vocab_size, output_size

    def forward(self, indices):
        assert torch.is_tensor(indices) and not torch.is_floating_point(indices), 'Use only tensor integer as indices, to avoid fancy indexing surprises'
        z = self.weight[indices]
        return z

    def __repr__(self):
        return f'Embedding({self.input_size}, {self.output_size}, bias=false)'


class BatchNorm(Module):

    def __init__(self, size, device='cpu'):
        self.beta = Param(1, size, init=init.zeros, device=device, requires_grad=True)
        self.gamma = Param(1, size, init=init.ones, device=device, requires_grad=True)

        self.running_mean = torch.zeros(1, size, device=device)
        self.running_var = torch.ones(1, size, device=device)
        self.decay = 0.9

    def forward(self, x):
        # mini-batch statistics
        if torch.is_grad_enabled():
            assert len(x) > 1, 'BatchNorm layer requires at least 2 samples in batch'

            mu, var = x.mean(dim=0), x.var(dim=0)
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * mu
            self.running_var  = self.decay * self.running_var  + (1 - self.decay) * var
        else:
            mu, var = self.running_mean, self.running_var

        # normalize x along the mini-batch
        x = (x - mu) / (var + 1e-5).sqrt()
        x = self.gamma * x + self.beta

        return x


class Dropout(Module):

    def __init__(self, p=.5):  # as prob to be zeroed
        assert 0 <= p < 1, f'Dropout probability must be in [0, 1], but got {p}'
        self.p = p

    def forward(self, x):  # note that each sample in the mini-batch is zeroed independently
        if torch.is_grad_enabled():
            x = x.clone()
            dropped = torch.rand_like(x) < self.p  # same as torch.bernoulli(x, self.p)
            x[dropped] = 0
            x /= (1 - self.p)  # This ensures that for any hidden unit the expected output (under the distribution used to drop units at training time) is the same as the actual output at test time

        return x



class RNN_cell(Module):

    def __init__(self, input_size, hidden_size, device='cpu'):
        self.embed = Embedding(input_size, hidden_size, device=device)  # no bias
        self.hidden = Linear(hidden_size, hidden_size, device=device, weights_init=init.xavier_normal)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, h=None):  # todo: support one-hot/dense input
        assert len(x.shape) == 1, 'x must be a 1D tensor (batch_size,)'
        N = x.shape

        if h is None:
            h = torch.zeros(self.hidden_size, device=self.device)

        xh = self.embed.forward(x)  # directly select the column embedding
        hh = self.hidden.forward(h)
        h = tanh(xh + hh)

        return h

    def __repr__(self):
        return f'RNN_cell(input_size={self.input_size}, hidden_size={self.hidden_size}): {self.n_params} params'


class RNN(Module):

    def __init__(self, input_size, hidden_size, backward=False, device='cpu'):
        self.rnn = RNN_cell(input_size, hidden_size, device=device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.backward = backward

    def forward(self, x, h=None):  # todo: support one-hot/dense input
        N, T = x.shape

        direction = reversed(range(T)) if self.backward else range(T)
        z = torch.zeros(N, T, self.hidden_size, device=self.device)
        for t in direction:
            h = self.rnn.forward(x[:, t], h)
            z[:, t] = h

        return z, h  # h == z[:, -1 or 0]  (i.e. the final hidden state for each batch element)

    def expression(self):
        direction = 't+1' if self.backward else 't-1'
        latex = r'$h_t = \tanh(W_{xh} x + W_{hh} h_{' + direction + r'} + b_h)$' + '\n'
        return latex

    def __repr__(self):
        return f'RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, backward={self.backward}): {self.n_params} params'
