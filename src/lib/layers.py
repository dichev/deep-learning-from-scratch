import torch
from lib.functions import init
from lib.functions.activations import tanh, sigmoid
from lib.base import Param, Module

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
        self.weight = Param(vocab_size, output_size, init=init.xavier_normal, device=device, requires_grad=True)
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


class LayerNorm(Module):

    def __init__(self, size, device='cpu'):
        self.shift = Param(1, size, init=init.zeros, device=device, requires_grad=True)
        self.scale = Param(1, size, init=init.ones, device=device, requires_grad=True)

    def forward(self, a):  # "a" are all pre-activations of the layer
        mu, var = a.mean(dim=-1, keepdim=True), a.var(dim=-1, keepdim=True)
        a = (a - mu) / (var + 1e-5).sqrt()
        a = self.scale * a + self.shift
        return a


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

    def __init__(self, input_size, hidden_size, layer_norm=False, device='cpu'):
        self.embed = Embedding(input_size, hidden_size, device=device)  # no bias
        self.hidden = Linear(hidden_size, hidden_size, device=device, weights_init=init.xavier_normal)
        if layer_norm:
            self.norm = LayerNorm(hidden_size, device=device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.device = device

    def forward(self, x, state=(None, None)):  # todo: support one-hot/dense input
        assert len(x.shape) == 1, 'x must be a 1D tensor (batch_size,)'
        N = x.shape[0]

        if state is None or state == (None, None):
            h = torch.zeros(N, self.hidden_size, device=self.device)
        else:
            h, _ = state

        xh = self.embed.forward(x)   # (N, F) -> (N, H)   directly select the column embedding
        hh = self.hidden.forward(h)  # (N, H) -> (N, H)
        a = xh + hh                  # (N, H) # todo: concatenate
        if self.layer_norm:
            a = self.norm.forward(a)
        h = tanh(a)

        return h, None

    def __repr__(self):
        return f'RNN_cell(input_size={self.input_size}, hidden_size={self.hidden_size}): {self.n_params} params'


class LSTM_cell(Module):
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.embed = Embedding(input_size, 4 * hidden_size, device=device)  # no bias
        self.gates = Linear(hidden_size, 4 * hidden_size, device=device, weights_init=init.xavier_normal)

        self._slice_i = slice(hidden_size * 0, hidden_size * 1)
        self._slice_f = slice(hidden_size * 1, hidden_size * 2)
        self._slice_o = slice(hidden_size * 2, hidden_size * 3)
        self._slice_m = slice(hidden_size * 3, None)  # todo: I am not a gate

        with torch.no_grad():
            self.gates.bias[:, self._slice_f] = 1.5  # set the sigmoid threshold beyond 0.5 to reduce the vanishing gradient at early stages of training (https://proceedings.mlr.press/v37/jozefowicz15.pdf)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, state=(None, None)):  # todo: support one-hot/dense input
        assert len(x.shape) == 1, 'x must be a 1D tensor (batch_size,)'
        N = x.shape[0]

        if state is None or state == (None, None):
            h = torch.zeros(N, self.hidden_size, device=self.device)    # fast state
            mem = torch.zeros(N, self.hidden_size, device=self.device)  # slow state
        else:
            h, mem = state

        xh = self.embed.forward(x)  # (N, _) -> (N, 4H)
        hh = self.gates.forward(h)  # (N, H) -> (N, 4H)
        a = xh + hh                 # (N, 4H)

        input_gate  = sigmoid(a[:, self._slice_i])   # input gate       (N, H)
        forget_gate = sigmoid(a[:, self._slice_f])   # forget gate      (N, H)
        output_gate = sigmoid(a[:, self._slice_o])   # output gate      (N, H)
        mem_candidate =  tanh(a[:, self._slice_m])   # new cell state   (N, H)

        mem = forget_gate * mem + input_gate * mem_candidate    # (N, H)
        h = output_gate * tanh(mem)                             # (N, H)

        return h, mem

class RNN(Module):

    def __init__(self, input_size, hidden_size, cell='rnn', backward=False, layer_norm=False, device='cpu'):
        if cell == 'rnn':
            self.cell = RNN_cell(input_size, hidden_size, layer_norm, device=device)
        elif cell == 'lstm':
            assert not layer_norm, 'LayerNorm is not supported for LSTM'
            self.cell = LSTM_cell(input_size, hidden_size, device=device)
        else:
            raise ValueError(f'Unknown cell type {cell}')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.backward = backward

    def forward(self, x, state=None):  # todo: support one-hot/dense input
        N, T = x.shape

        direction = reversed(range(T)) if self.backward else range(T)
        z = torch.zeros(N, T, self.hidden_size, device=self.device)
        for t in direction:
            state = self.cell.forward(x[:, t], state)
            h, _ = state
            z[:, t] = h

        return z, state  # h == z[:, -1 or 0]  (i.e. the final hidden state for each batch element)

    def expression(self):
        direction = 't+1' if self.backward else 't-1'
        latex = r'$h_t = \tanh(W_{xh} x + W_{hh} h_{' + direction + r'} + b_h)$' + '\n'
        return latex

    def __repr__(self):
        return f'RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, backward={self.backward}): {self.n_params} params'
