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
        self.linear = Linear(input_size + hidden_size, hidden_size, device=device, weights_init=init.xavier_normal)  # (I+H, H)
        if layer_norm:
            self.norm = LayerNorm(hidden_size, device=device)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.device = device

    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape

        h, _ = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=self.device)

        xh = torch.concat((x, h), dim=-1)  # (N, I+H)
        a = self.linear.forward(xh)                # (N, I+H) -> (N, H)
        if self.layer_norm:
            a = self.norm.forward(a)               # (N, H)
        h = tanh(a)                                # (N, H)

        return h, None

    def __repr__(self):
        return f'RNN_cell(input_size={self.input_size}, hidden_size={self.hidden_size}): {self.n_params} params'


class LSTM_cell(Module):
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.linear = Linear(input_size + hidden_size, 4 * hidden_size, device=device, weights_init=init.xavier_normal)  # (I+H, 4H)

        self._slice_i = slice(hidden_size * 0, hidden_size * 1)
        self._slice_f = slice(hidden_size * 1, hidden_size * 2)
        self._slice_o = slice(hidden_size * 2, hidden_size * 3)
        self._slice_m = slice(hidden_size * 3, None)  # todo: I am not a gate

        with torch.no_grad():
            self.linear.bias[:, self._slice_f] = 1.  # set the sigmoid threshold beyond 0.5 to reduce the vanishing gradient at early stages of training (https://proceedings.mlr.press/v37/jozefowicz15.pdf)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape

        h, mem = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=self.device)    # fast state
        if mem is None:
            mem = torch.zeros(N, self.hidden_size, device=self.device)  # slow state

        xh = torch.concat((x, h), dim=-1)  # (N, I), (N, H)  -> (N, I+H)
        a = self.linear.forward(xh)                # (N, I+H) -> (N, 4H)

        input_gate  = sigmoid(a[:, self._slice_i])   # input gate       (N, H)
        forget_gate = sigmoid(a[:, self._slice_f])   # forget gate      (N, H)
        output_gate = sigmoid(a[:, self._slice_o])   # output gate      (N, H)
        mem_candidate =  tanh(a[:, self._slice_m])   # new cell state   (N, H)

        mem = forget_gate * mem + input_gate * mem_candidate    # (N, H)
        h = output_gate * tanh(mem)                             # (N, H)

        return h, mem


class GRU_cell(Module):
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.gates = Linear(input_size + hidden_size, 2 * hidden_size, device=device, weights_init=init.xavier_normal)  # (I+H, 2H)
        self.candidate = Linear(input_size + hidden_size, hidden_size, device=device, weights_init=init.xavier_normal)  # (I+H, H)

        self._slice_r = slice(hidden_size * 0, hidden_size * 1)  # reset gate params
        self._slice_z = slice(hidden_size * 1, hidden_size * 2)  # update gate params

        # with torch.no_grad():
        #     self.linear.bias[:, self._slice_f] = 1.  # set the sigmoid threshold beyond 0.5 to reduce the vanishing gradient at early stages of training (https://proceedings.mlr.press/v37/jozefowicz15.pdf)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape
        h, _ = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=self.device)

        # Compute the gates
        xh = torch.concat((x, h), dim=-1)    # (N, I+H)
        a = self.gates.forward(xh)                   # (N, 2H)
        reset_gate  = sigmoid(a[:, self._slice_r])   # (N, H)
        update_gate = sigmoid(a[:, self._slice_z])   # (N, H)

        # Compute the candidate state
        xh_ = torch.concat((x, reset_gate * h), dim=-1)  # (N, I+H)
        mem_candidate = tanh(self.candidate.forward(xh_))        # (N, 2H)

        # Update the hidden state
        h = update_gate * h + (1-update_gate) * mem_candidate    # (N, H)

        return h, None


class RNN(Module):

    def __init__(self, input_size, hidden_size, cell='rnn', backward=False, layer_norm=False, device='cpu'):
        if cell == 'rnn':
            self.cell = RNN_cell(input_size, hidden_size, layer_norm, device=device)
        elif cell == 'lstm':
            assert not layer_norm, 'LayerNorm is not supported for LSTM'
            self.cell = LSTM_cell(input_size, hidden_size, device=device)
        elif cell == 'gru':
            assert not layer_norm, 'LayerNorm is not supported for GRU'
            self.cell = GRU_cell(input_size, hidden_size, device=device)
        else:
            raise ValueError(f'Unknown cell type {cell}')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.backward = backward

    def forward(self, x, state=None):
        assert len(x.shape) == 3, f'Expected 3D tensor (batch_size, time_steps, features) as input, but got {x.shape}'
        N, T, F = x.shape

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
