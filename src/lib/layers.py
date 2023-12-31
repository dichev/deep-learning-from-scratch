import torch
import torch.nn.functional as F
from lib.functions import init
from lib.functions.activations import tanh, sigmoid
from lib.base import Param, Module
from utils.other import conv2d_calc_out_size, conv2d_pad_string_to_int
from collections import namedtuple

Padding = namedtuple('Padding', ('pad_left', 'pad_right', 'pad_top', 'pad_bottom'))

class Linear(Module):
    def __init__(self, input_size, output_size=1, weights_init=init.kaiming_normal_relu_, device='cpu'):
        self.weight = Param((input_size, output_size), device=device)  # (D, C)
        self.bias = Param((1, output_size), device=device)  # (D, C)
        self.input_size, self.output_size = input_size, output_size
        self.weights_initializer = weights_init
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weights_initializer(self.weight, self.input_size, self.output_size)  # kaiming_normal_relu_ by default
        self.bias.fill_(0)

    def forward(self, X):
        z = X @ self.weight + self.bias    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z

    def __repr__(self):
        return f'Linear({self.input_size}, {self.output_size}, bias=true): {self.n_params} params'


class Embedding(Module):  # aka lookup table
    def __init__(self, vocab_size, output_size, padding_idx=None, device='cpu'):
        self.weight = Param((vocab_size, output_size), device=device)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx] = 0.

        self.input_size, self.output_size = vocab_size, output_size
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.kaiming_normal_relu_(self.weight, self.input_size)

    def forward(self, indices):
        assert torch.is_tensor(indices) and not torch.is_floating_point(indices), 'Use only tensor integer as indices, to avoid fancy indexing surprises'
        z = self.weight[indices]
        return z

    def __repr__(self):
        return f'Embedding({self.input_size}, {self.output_size}, bias=false): {self.n_params} params'


class BatchNorm(Module):

    def __init__(self, size, device='cpu'):
        self.beta = Param((1, size), device=device)
        self.gamma = Param((1, size), device=device)

        self.running_mean = torch.zeros(1, size, device=device)
        self.running_var = torch.ones(1, size, device=device)
        self.decay = 0.9
        self.size = size
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.beta.fill_(0)
        self.beta.fill_(1)

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

    def __repr__(self):
        return f'BatchNorm({self.size}): {self.n_params} params'


class LayerNorm(Module):

    def __init__(self, size, device='cpu'):
        self.shift = Param((1, size), device=device)
        self.scale = Param((1, size), device=device)
        self.size = size
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.shift.fill_(0)
        self.scale.fill_(1)

    def forward(self, a):  # "a" are all pre-activations of the layer
        mu, var = a.mean(dim=-1, keepdim=True), a.var(dim=-1, keepdim=True)
        a = (a - mu) / (var + 1e-5).sqrt()
        a = self.scale * a + self.shift
        return a

    def __repr__(self):
        return f'LayerNorm({self.size}): {self.n_params} params'


class LocalResponseNorm(Module):  # Inter-channel: https://miro.medium.com/v2/resize:fit:720/format:webp/1*MFl0tPjwvc49HirAJZPhEA.png

    def __init__(self, size, alpha=1e-4, beta=.75, k=1.):
        assert size % 2 == 1, f'size must be odd, but got {size}'
        self.size = size    # neighborhood length
        self.alpha = alpha  # scale factor - used as alpha/size factor, to make the hyperparameter α less sensitive to different sizes
        self.beta = beta    # exponent
        self.k = k          # bias, avoids division by zero

    def forward(self, x):
        B, C, W, H = x.shape
        n = self.size

        a_sq = (x**2).view(B, 1, C, W * H)  # square earlier and adapt the shape for unfolding
        a_sq = F.unfold(a_sq, kernel_size=(n, 1), padding=(n//2, 0))    # (B, window, patches)
        a_sq_sum = a_sq.view(B, n, C, W, H).sum(dim=1)          # (B, C  W, H)
        x = x / (self.k + (self.alpha / n) * a_sq_sum) ** self.beta     # (B, C, W, H)
        return x

    def __repr__(self):
        return f'LocalResponseNorm({self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k})'


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

    def __repr__(self):
        return f'Dropout({self.p})'

class RNN_cell(Module):

    def __init__(self, input_size, hidden_size, layer_norm=False, device='cpu'):
        self.weight = Param((input_size + hidden_size, hidden_size), device=device)  # (I+H, H)
        self.bias = Param((1, hidden_size), device=device)  # (1, H)
        if layer_norm:
            self.norm = LayerNorm(hidden_size, device=device)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.device = device
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.xavier_normal_(self.weight, self.weight.shape[0], self.weight.shape[1])
        self.bias.fill_(1)

    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape
        h, _ = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=self.device)

        # Compute the hidden state
        xh = torch.concat((x, h), dim=-1)  # (N, I+H)
        a = xh @ self.weight + self.bias           # (N, I+H) -> (N, H)
        if self.layer_norm:
            a = self.norm.forward(a)               # (N, H)
        h = tanh(a)                                # (N, H)

        return h, None

    def __repr__(self):
        return f'RNN_cell({self.input_size}, {self.hidden_size}, layer_norm={self.layer_norm}): {self.n_params} params'


class LSTM_cell(Module):
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.weight = Param((input_size + hidden_size, 4 * hidden_size), device=device)  # (I+H, 4H)
        self.bias = Param((1, 4 * hidden_size), device=device)  # (1, 4H)

        self._slice_i = slice(hidden_size * 0, hidden_size * 1)
        self._slice_f = slice(hidden_size * 1, hidden_size * 2)
        self._slice_o = slice(hidden_size * 2, hidden_size * 3)
        self._slice_m = slice(hidden_size * 3, None)

        with torch.no_grad():
            self.bias[:, self._slice_f] = 1.  # set the sigmoid threshold beyond 0.5 to reduce the vanishing gradient at early stages of training (https://proceedings.mlr.press/v37/jozefowicz15.pdf)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.xavier_normal_(self.weight, self.weight.shape[0], self.weight.shape[1])
        self.bias.fill_(0)

    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape
        h, mem = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=self.device)    # fast state
        if mem is None:
            mem = torch.zeros(N, self.hidden_size, device=self.device)  # slow state

        # Compute the gates
        xh = torch.concat((x, h), dim=-1)    # (N, I), (N, H)  -> (N, I+H)
        a = xh @ self.weight + self.bias             # (N, I+H) -> (N, 4H)
        input_gate  = sigmoid(a[:, self._slice_i])   # input gate       (N, H)
        forget_gate = sigmoid(a[:, self._slice_f])   # forget gate      (N, H)
        output_gate = sigmoid(a[:, self._slice_o])   # output gate      (N, H)
        mem_candidate =  tanh(a[:, self._slice_m])   # new cell state   (N, H)

        # Update the hidden and memory state
        mem = forget_gate * mem + input_gate * mem_candidate    # (N, H)
        h = output_gate * tanh(mem)                             # (N, H)

        return h, mem

    def __repr__(self):
        return f'LSTM_cell({self.input_size}, {self.hidden_size}): {self.n_params} params'


class GRU_cell(Module):
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.weight = Param((input_size + hidden_size, 3 * hidden_size), device=device)  # (I+H, 3H)
        self.bias = Param((1, 3 * hidden_size), device=device)  # (1, 3H)

        self._slice_r = slice(hidden_size * 0, hidden_size * 1)  # reset gate params
        self._slice_z = slice(hidden_size * 1, hidden_size * 2)  # update gate params
        self._slice_n = slice(hidden_size * 1, hidden_size * 2)  # new cell candidate params

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.xavier_normal_(self.weight, self.weight.shape[0], self.weight.shape[1])
        self.bias.fill_(0)

    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape
        h, _ = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=self.device)

        # Just references
        W_xr, W_xz, W_xn = (self.weight[:, _slice] for _slice in [self._slice_r, self._slice_z, self._slice_n])
        b_xr, b_xz, b_xn = (self.bias[:, _slice] for _slice in [self._slice_r, self._slice_z, self._slice_n])

        # Compute the gates
        xh = torch.concat((x, h), dim=-1)  # (N, I+H)
        reset_gate  = sigmoid(xh @ W_xr + b_xr)    # (N, H)
        update_gate = sigmoid(xh @ W_xz + b_xz)    # (N, H) # note: reset and update gates computation can be combined, but is not for clarity

        # Compute the candidate state
        xh_ = torch.concat((x, reset_gate * h), dim=-1)  # (N, I+H)
        mem_candidate = tanh(xh_ @ W_xn + b_xn)                  # (N, 2H)

        # Update the hidden state
        h = update_gate * h + (1-update_gate) * mem_candidate    # (N, H)

        return h, None

    def __repr__(self):
        return f'GRU_cell({self.input_size}, {self.hidden_size}): {self.n_params} params'

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
        self.layer_norm = layer_norm
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.cell.reset_parameters()

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
        return f'RNN({self.input_size}, {self.hidden_size}, {self.cell}, backward={self.backward}, layer_norm={self.layer_norm}): {self.n_params} params'


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, device='cpu'):
        self.weight = Param((out_channels, in_channels, kernel_size, kernel_size), device=device)  # (C_out, C_in, K, K)
        self.bias = Param((out_channels,), device=device)  # (C)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.device = device
        self.padding = padding
        if isinstance(padding, str):  # e.g. valid, same, full
            self.padding = conv2d_pad_string_to_int(padding, kernel_size)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.kaiming_normal_relu_(self.weight, self.in_channels * self.kernel_size * self.kernel_size)
        init.kaiming_normal_relu_(self.bias, 1)

    def forward(self, X):
        N, C, W, H, = X.shape
        W_out, H_out = conv2d_calc_out_size(X, self.kernel_size, self.stride, self.padding, self.dilation)  # useful validation

        """
        # cross-correlation between batch images and filters:
        k = self.kernel_size
        Y = torch.zeros((N, self.out_channels, out_size, out_size), device=self.device)  # (N, C_out, W_out, H_out)
        for h in range(out_size):
            for w in range(out_size):
                for c in range(self.out_channels):
                    patches = X[:, :, w:w+k, h:h+k].reshape(N, -1)
                    kernel = self.weight[c].flatten()
                    bias = self.bias[c]
                    Y[:, c, w, h] = patches @ kernel + bias
        """

        # Vectorized batched convolution: Y = [I] * K + b  (which is actually cross-correlation + bias shifting)
        patches = F.unfold(X, self.kernel_size, self.dilation, self.padding, self.stride)                            # (N, kernel_size_flat, patches)
        kernel = self.weight.reshape(self.out_channels, -1)                                                          # * (channels, kernel_size_flat)
        convolution = torch.einsum('nkp,ck->ncp', patches, kernel)                                            # -> (N, channels, patches)
        Y = convolution.reshape(N, self.out_channels, W_out, H_out) + self.bias.reshape(1, -1, 1, 1)   # (N, channels, out_width, out_height)

        return Y  # (N, C_out, W_out, H_out)

    def __repr__(self):
        return f'Conv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}): {self.n_params} parameters'


class Pool2d(Module):

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, device='cpu', padding_fill_value=0.):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.device = device
        self.padding_fill_value = padding_fill_value

        if padding in ('valid', 'same', 'full'):
            padding = conv2d_pad_string_to_int(padding, kernel_size)

        if isinstance(padding, int):
            assert padding <= kernel_size // 2, f'Padding should be at most half of kernel size, but got padding={padding}, kernel_size={kernel_size}'
            self.padding = Padding(padding, padding, padding, padding)
        else:  # tuple
            assert len(padding) == 4, f'Expected padding = (left, right, top, bottom), but got {padding}'
            assert min(padding) <= kernel_size // 2, f'Padding should be at most half of kernel size, but got padding={padding}, kernel_size={kernel_size}'
            self.padding = Padding(*padding)

    def forward(self, X):
        N, C, W, H, = X.shape
        W_out, H_out = conv2d_calc_out_size(X, self.kernel_size, self.stride, self.padding, self.dilation)  # useful validation

        if max(self.padding) > 0:  # the padding value for max pooling must be inf negatives for correct max pooling of negative inputs
            X = F.pad(X, self.padding, mode='constant', value=self.padding_fill_value)

        # Vectorized batched max pooling: Y = max[I]
        patches = F.unfold(X, self.kernel_size, self.dilation, padding=0, stride=self.stride)             # (N, kernel_size_flat, patches)
        patches = patches.reshape(N, C, self.kernel_size*self.kernel_size, -1)                    # (N, C, kernel_size_flat, patches)
        pooled = self.pool(patches)                                                                       # (N, C, patches)
        Y = pooled.reshape(N, C, W_out, H_out)                                                            # (N, C, W_out, H_out)

        return Y  # (N, C, W_out, H_out)

    def pool(self, patches):
        raise Exception('Not implemented')


class MaxPool2d(Pool2d):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, device='cpu'):
        super().__init__(kernel_size, stride, padding, dilation, device, padding_fill_value=-torch.inf)  # use padding with inf negatives for correct max pooling of negative inputs

    def pool(self, patches):
        max_pooled, _ = patches.max(dim=2)
        return max_pooled

    def __repr__(self):
        return f'MaxPool2d({self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'


class AvgPool2d(Pool2d):
    def pool(self, patches):
        return patches.mean(dim=2)

    def __repr__(self):
        return f'AvgPool2d({self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'


class Sequential(Module):
    def __init__(self, *modules):
        self._steps = []
        for i, module in enumerate(modules):
            self.add(module)

    def add(self, module):
        name = f'm{len(self._steps)}'
        setattr(self, name, module)
        self._steps.append(module)

    def forward(self, x, verbose=False):
        if verbose:
            print(list(x.shape), 'Input')
        for module in self._steps:
            if isinstance(module, Sequential):
                x = module.forward(x, verbose=verbose)
            elif isinstance(module, Module):
                x = module.forward(x)
            elif callable(module):
                x = module(x)
            else:
                raise Exception('Unexpected module: ' + type(module))
            if verbose:
                print(list(x.shape), module)
        # for name, module in self.modules():
        #     x = module.forward(x)
        return x
