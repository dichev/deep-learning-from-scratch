import torch
import torch.nn.functional as F
import einops as ein
import warnings
from math import sqrt
from matplotlib import pyplot as plt
from lib.functions import init
from lib.functions.activations import relu, gelu, gelu_tanh_approx, tanh, sigmoid, softmax, swish
from lib.functions.losses import entropy
from lib.base import Param, Module, ModuleList, Sequential
from utils.other import conv2d_calc_out_size, conv2d_pad_string_to_int, identity
from utils import plots
from collections import namedtuple

Padding = namedtuple('Padding', ('pad_left', 'pad_right', 'pad_top', 'pad_bottom'))

class Linear(Module):
    def __init__(self, input_size, output_size=1, weights_init=init.linear_uniform_, bias=True):
        self.weight = Param((input_size, output_size))
        self.bias = Param((1, output_size)) if bias else 0.
        self.input_size, self.output_size = input_size, output_size
        self.weights_initializer = weights_init
        self.has_bias = bias
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weights_initializer(self.weight, self.input_size, self.output_size)
        if self.has_bias:
            self.weights_initializer(self.bias, self.input_size, self.output_size)

    def forward(self, X):
        z = X @ self.weight + self.bias    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z

    def __repr__(self):
        return f'Linear({self.input_size}, {self.output_size}, bias={self.has_bias}): {self.n_params} params'


class Embedding(Module):  # aka lookup table
    def __init__(self, vocab_size, embed_size, padding_idx=None):
        self.weight = Param((vocab_size, embed_size))
        self.padding_idx = padding_idx
        self.input_size, self.output_size = vocab_size, embed_size
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weight.normal_()
        if self.padding_idx is not None:
            self.weight[self.padding_idx] = 0.

    def forward(self, indices):
        assert torch.is_tensor(indices) and not torch.is_floating_point(indices), 'Use only tensor integer as indices, to avoid fancy indexing surprises'
        z = self.weight[indices]
        return z

    def backward(self, x):  # used for tied embedding to output linear
        z = x @ self.weight.T
        return z

    def __repr__(self):
        return f'Embedding({self.input_size}, {self.output_size}, bias=false): {self.n_params} params'



class BatchNorm(Module):
    """
    Paper: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
    https://proceedings.mlr.press/v37/ioffe15.pdf
    """

    def __init__(self, size, batch_dims=()):
        shape = tuple(size if d not in batch_dims else 1 for d in range(len(batch_dims) + 1))  # necessary to maintain proper broadcasting
        self.beta = Param(shape)
        self.gamma = Param(shape)

        self.running_mean = self.register_buffer('running_mean', torch.zeros(shape))
        self.running_var = self.register_buffer('running_var', torch.ones(shape))
        self.decay = 0.9
        self.size = size
        self.dims = batch_dims
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.beta.fill_(0)
        self.gamma.fill_(1)
        self.running_mean.fill_(0)
        self.running_var.fill_(1)

    def forward(self, x):
        assert len(x.shape) == len(self.dims) + 1, f'Expect tensor with {len(self.dims) + 1} axis, but got {x.shape}'

        # mini-batch statistics
        if torch.is_grad_enabled():
            mu, var, var_unbias = x.mean(dim=self.dims, keepdims=True), x.var(dim=self.dims, correction=0, keepdims=True), x.var(dim=self.dims, keepdims=True)
            with torch.no_grad():  # fix a memory leak in the computation graph (for var_unbias only)
                self.running_mean[:] = self.decay * self.running_mean + (1 - self.decay) * mu
                self.running_var[:]  = self.decay * self.running_var  + (1 - self.decay) * var_unbias
        else:
            mu, var = self.running_mean, self.running_var

        # normalize x along the mini-batch
        x = (x - mu) / (var + 1e-5).sqrt()
        x = self.gamma * x + self.beta

        return x

    def __repr__(self):
        return f'{self.__class__.__name__}({self.size}, batch_dims={self.dims}): {self.n_params} params'

class BatchNorm1d(BatchNorm):

    def __init__(self, size):
        super().__init__(size, batch_dims=(0,))

    def forward(self, x):
        assert len(x.shape) == 2, f'Linear layer expect 2d tensor (batch, features), but got {x.shape}'
        assert len(x) > 1 or not torch.is_grad_enabled(), 'BatchNorm1d layer requires at least 2 samples in batch'
        return super().forward(x)


class BatchNorm2d(BatchNorm):

    def __init__(self, size):
        super().__init__(size, batch_dims=(0, 2, 3))  # note here N, H, W are all considered to be from the batch dim

    def forward(self, x):
        assert len(x.shape) == 4, f'BatchNorm2d layer requires 4D input, but got: {x.shape}'
        N, C, H, W = x.shape
        assert C == self.size, f'Expected {self.size} channels from the input, but got {C}'
        assert max(N, H, W) > 1 or not torch.is_grad_enabled(), 'BatchNorm2d layer requires at least 2 samples'
        return super().forward(x)


class LayerNorm(Module):
    """
    Paper: Layer Normalization
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, size, eps=1e-5):
        self.shift = Param((1, size))
        self.scale = Param((1, size))
        self.size = size
        self.eps = eps
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.shift.fill_(0)
        self.scale.fill_(1)

    def forward(self, a):  # "a" are all pre-activations of the layer
        mu, var = a.mean(dim=-1, keepdim=True), a.var(dim=-1, keepdim=True, correction=0)
        a = (a - mu) / (var + self.eps).sqrt()
        a = self.scale * a + self.shift
        return a

    def __repr__(self):
        return f'LayerNorm({self.size}): {self.n_params} params'


class RMSNorm(Module):
    """
    Paper: Root Mean Square Layer Normalization
    https://arxiv.org/pdf/1910.07467
    """

    def __init__(self, size, eps=1e-6, partial=1.):
        self.gain = Param((1, size))
        self.size = size
        self.eps = eps
        self.p = partial
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.gain.fill_(1)

    def forward(self, a):
        if self.p == 1.:
            rms = a.norm(dim=-1, keepdim=True) * (1/self.size)**0.5   # equivalent to: (a**2).mean().sqrt()
        else:  # partial sample statistic
            sample_size = round(self.p*self.size)
            rms = a[..., :sample_size].norm(dim=-1, keepdim=True) * (1/sample_size)**0.5

        a = a / (rms + self.eps) * self.gain
        return a

    def __repr__(self):
        return f'RMSNorm({self.size}, partial={self.p}): {self.n_params} params'



class LocalResponseNorm(Module):  # Inter-channel: https://miro.medium.com/v2/resize:fit:720/format:webp/1*MFl0tPjwvc49HirAJZPhEA.png
    """
    Paper: ImageNet Classification with Deep Convolutional Neural Networks
    https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """

    def __init__(self, size, alpha=1e-4, beta=.75, k=1.):
        assert size % 2 == 1, f'size must be odd, but got {size}'
        self.size = size    # neighborhood length
        self.alpha = alpha  # scale factor - used as alpha/size factor, to make the hyperparameter α less sensitive to different sizes
        self.beta = beta    # exponent
        self.k = k          # bias, avoids division by zero

    def forward(self, x):
        B, C, H, W = x.shape
        n = self.size

        a_sq = (x**2).view(B, 1, C, W * H)  # square earlier and adapt the shape for unfolding
        a_sq = F.unfold(a_sq, kernel_size=(n, 1), padding=(n//2, 0))    # (B, window, patches)
        a_sq_sum = a_sq.view(B, n, C, H, W).sum(dim=1)                  # (B, C  H, W)
        x = x / (self.k + (self.alpha / n) * a_sq_sum) ** self.beta     # (B, C, H, W)
        return x

    def __repr__(self):
        return f'LocalResponseNorm({self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k})'


class Dropout(Module):
    """
    Paper: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """

    def __init__(self, p=.5):  # as prob to be zeroed
        assert 0 <= p < 1, f'Dropout probability must be in [0, 1), but got {p}'
        self.p = p

    def forward(self, x):  # note that each sample in the mini-batch is zeroed independently
        if self.p > 0 and torch.is_grad_enabled():
            dropped = torch.rand_like(x) < self.p  # same as torch.bernoulli(x, self.p)
            x = x.masked_fill(dropped, 0)
            x /= (1 - self.p)  # This ensures that for any hidden unit the expected output (under the distribution used to drop units at training time) is the same as the actual output at test time

        return x

    def __repr__(self):
        return f'Dropout({self.p})'


class RNN_cell(Module):

    def __init__(self, input_size, hidden_size, layer_norm=False, use_relu=False):
        self.weight = Param((input_size + hidden_size, hidden_size))  # (I+H, H)
        self.bias = Param((1, hidden_size))  # (1, H)
        if layer_norm:
            self.norm = LayerNorm(hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.use_relu = use_relu
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.use_relu:
            init.kaiming_normal_relu_(self.weight, self.weight.shape[0])
            self.bias.fill_(0)
        else:
            init.xavier_normal_(self.weight, self.weight.shape[0], self.weight.shape[1])
            self.bias.fill_(1)

    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape
        h, _ = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=x.device)

        # Compute the hidden state
        xh = torch.concat((x, h), dim=-1)      # (N, I+H)
        a = xh @ self.weight + self.bias               # (N, I+H) -> (N, H)

        if self.layer_norm:
            a = self.norm.forward(a)                   # (N, H)

        h = tanh(a) if not self.use_relu else relu(a)  # (N, H)

        return h, None

    def __repr__(self):
        return f'RNN_cell({self.input_size}, {self.hidden_size}, layer_norm={self.layer_norm}): {self.n_params} params'


class LSTM_cell(Module):
    """
    Paper: Generating Sequences With Recurrent Neural Networks
    https://arxiv.org/pdf/1308.0850.pdf
    """
    def __init__(self, input_size, hidden_size):
        self.weight = Param((input_size + hidden_size, 4 * hidden_size))  # (I+H, 4H)
        self.bias = Param((1, 4 * hidden_size))  # (1, 4H)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.xavier_normal_(self.weight, self.weight.shape[0], self.weight.shape[1])
        self.bias.fill_(0)
        # set the sigmoid threshold (of forget gate) beyond 0.5 to reduce the vanishing gradient at early stages of training (https://proceedings.mlr.press/v37/jozefowicz15.pdf)
        self.bias[:, self.hidden_size:self.hidden_size*2] = 1.


    def forward(self, x, state=(None, None)):
        assert len(x.shape) == 2, f'Expected (batch_size, features) as input, got {x.shape}'
        N, F = x.shape
        h, mem = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=x.device)    # fast state
        if mem is None:
            mem = torch.zeros(N, self.hidden_size, device=x.device)  # slow state

        # Compute the gates
        xh = torch.concat((x, h), dim=-1)    # (N, I), (N, H)  -> (N, I+H)
        a = xh @ self.weight + self.bias             # (N, I+H) -> (N, 4H)

        i, f, o, m = a.chunk(4, dim=-1)     # (N, H) each
        input_gate  = sigmoid(i)   # input gate       (N, H)
        forget_gate = sigmoid(f)   # forget gate      (N, H)
        output_gate = sigmoid(o)   # output gate      (N, H)
        mem_candidate =  tanh(m)   # new cell state   (N, H)

        # Update the hidden and memory state
        mem = forget_gate * mem + input_gate * mem_candidate    # (N, H)
        h = output_gate * tanh(mem)                             # (N, H)

        return h, mem

    def __repr__(self):
        return f'LSTM_cell({self.input_size}, {self.hidden_size}): {self.n_params} params'


class GRU_cell(Module):
    def __init__(self, input_size, hidden_size):
        self.weight = Param((input_size + hidden_size, 3 * hidden_size))  # (I+H, 3H)
        self.bias = Param((1, 3 * hidden_size))  # (1, 3H)
        self.input_size = input_size
        self.hidden_size = hidden_size
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
            h = torch.zeros(N, self.hidden_size, device=x.device)

        # Just references
        W_xr, W_xz, W_xn = self.weight.chunk(3, dim=-1)
        b_xr, b_xz, b_xn = self.bias.chunk(3, dim=-1)

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

    def __init__(self, input_size, hidden_size, cell='rnn', backward=False, layer_norm=False):
        if cell == 'rnn':
            self.cell = RNN_cell(input_size, hidden_size, layer_norm)
        elif cell == 'lstm':
            assert not layer_norm, 'LayerNorm is not supported for LSTM'
            self.cell = LSTM_cell(input_size, hidden_size)
        elif cell == 'gru':
            assert not layer_norm, 'LayerNorm is not supported for GRU'
            self.cell = GRU_cell(input_size, hidden_size)
        else:
            raise ValueError(f'Unknown cell type {cell}')
        self.input_size = input_size
        self.hidden_size = hidden_size
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
        z = torch.zeros(N, T, self.hidden_size, device=x.device)
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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, mem_optimized=False):
        self.weight = Param((out_channels, in_channels, kernel_size, kernel_size))  # (C_out, C_in, K, K)
        if bias:
            self.bias = Param((out_channels,))  # (C)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.mem_optimized = mem_optimized
        if isinstance(padding, str):  # e.g. valid, same, full
            self.padding = conv2d_pad_string_to_int(padding, kernel_size)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        init.linear_uniform_(self.weight, fan_in)
        if self.has_bias:
            init.linear_uniform_(self.bias, fan_in)

    def forward(self, X):
        if self.mem_optimized:  # Use torch to reduce memory usage on large models
            return torch.nn.functional.conv2d(X, self.weight, self.bias if self.has_bias else None, stride=self.stride, padding=self.padding, dilation=self.dilation)

        N, C, H, W = X.shape
        H_out, W_out = conv2d_calc_out_size(X, self.kernel_size, self.stride, self.padding, self.dilation)  # useful validation

        # Vectorized batched convolution: Y = [I] * K + b  (which is actually cross-correlation + bias shifting)
        patches = F.unfold(X, self.kernel_size, self.dilation, self.padding, self.stride)  # (N, kernel_size_flat, patches)
        kernel = self.weight.reshape(self.out_channels, -1)                                # * (channels, kernel_size_flat)
        convolution = torch.einsum('nkp,ck->ncp', patches, kernel)                         # -> (N, channels, patches)
        Y = convolution.reshape(N, self.out_channels, H_out, W_out)                        # (N, channels, out_width, out_height)
        if self.has_bias:
            Y += self.bias.reshape(1, -1, 1, 1)

        return Y  # (N, C_out, H_out, W_out)

    def __repr__(self):
        return f'Conv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.has_bias}): {self.n_params} parameters'



class ConvTranspose2d(Module):

    # Matches the arguments (padding, kernel, etc.) to Conv2d, except in_channels and out_channels which are flipped
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, mem_optimized=False):
        self.weight = Param((in_channels, out_channels, kernel_size, kernel_size))  # (C_in, C_out, K, K)
        if bias:
            self.bias = Param((out_channels,))  # (C)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.mem_optimized = mem_optimized
        if isinstance(padding, str):  # e.g. valid, same, full
            self.padding = conv2d_pad_string_to_int(padding, kernel_size)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        fan_in = self.out_channels * self.kernel_size * self.kernel_size  # note: out_channels is used to match pytorch's initialization
        init.linear_uniform_(self.weight, fan_in)
        if self.has_bias:
            init.linear_uniform_(self.bias, fan_in)

    def forward(self, Y):
        if self.mem_optimized:  # Use torch to reduce memory usage on large models
            return torch.nn.functional.conv_transpose2d(Y, self.weight, self.bias if self.has_bias else None, stride=self.stride, padding=self.padding, dilation=self.dilation)

        N, C, H, W = Y.shape
        H_out, W_out = conv2d_calc_out_size(Y, self.kernel_size, self.stride, self.padding, self.dilation, transposed=True)  # useful validation

        # basically reverse the steps of the convolution operations in Conv2d (not true inverse)
        convolution = Y.view(N, self.in_channels, -1)
        kernel = self.weight.reshape(self.in_channels, -1)
        patches = torch.einsum('ncp,ck->nkp', convolution, kernel)
        X = F.fold(patches, output_size=(H_out, W_out), kernel_size=self.kernel_size,  dilation=self.dilation, padding=self.padding, stride=self.stride)
        if self.has_bias: # notice the bias is added to X (in convolution operator it was added to Y)
            X += self.bias.reshape(1, -1, 1, 1)

        return X  # (N, C_out, H_out, W_out)

    def __repr__(self):
        return f'ConvTranspose2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.has_bias}): {self.n_params} parameters'



class Conv2dGroups(Module):  # implemented as a stack of convolutional layers
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        assert in_channels % groups == 0 and out_channels % groups == 0, f'the channels must be divisible by the groups, but got: {in_channels=}, {out_channels=} for {groups=}'
        self.convs = ModuleList(
            Conv2d(in_channels//groups, out_channels//groups, kernel_size, stride, padding, dilation, bias)
            for g in range(groups)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        if isinstance(padding, str):  # e.g. valid, same, full
            self.padding = conv2d_pad_string_to_int(padding, kernel_size)
        self.groups = groups

    def forward(self, X):
        N, C, H, W = X.shape
        H_out, W_out = conv2d_calc_out_size(X, self.kernel_size, self.stride, self.padding, self.dilation)  # useful validation

        Y = torch.zeros(N, self.out_channels, H_out, W_out, device=X.device)
        for g in range(self.groups):
            Y[:, self.slice_out(g)] = self.convs[g].forward(X[:, self.slice_in(g)])
        return Y

    def slice_in(self, group):
        step = self.in_channels // self.groups
        return slice(group * step, (group + 1) * step)

    def slice_out(self, group):
        step = self.out_channels // self.groups
        return slice(group * step, (group + 1) * step)

    def __repr__(self):
        return f'Conv2dGroup({self.in_channels}, {self.out_channels}, {self.kernel_size}, groups={self.groups} stride={self.stride}, padding={self.padding}, dilation={self.dilation}): {self.n_params} parameters'

class Pool2d(Module):

    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, padding_fill_value=0.):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
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
        N, C, H, W = X.shape
        if self.kernel_size == W == H and self.stride == 1 and self.dilation == 1 and self.padding == (0, 0, 0, 0):
            # shortcut computations if the pooling is global (but still channel-wise)
            return self.pool(X.view(N, C, -1)).view(N, C, 1, 1)       # (N, C, H, W) -> # (N, C, 1, 1)

        H_out, W_out = conv2d_calc_out_size(X, self.kernel_size, self.stride, self.padding, self.dilation)  # useful validation

        if max(self.padding) > 0:  # the padding value for max pooling must be inf negatives for correct max pooling of negative inputs
            X = F.pad(X, self.padding, mode='constant', value=self.padding_fill_value)

        # Vectorized batched max pooling: Y = max[I]
        patches = F.unfold(X, self.kernel_size, self.dilation, padding=0, stride=self.stride)     # (N, kernel_size_flat, patches)
        patches = patches.reshape(N, C, self.kernel_size*self.kernel_size, -1)                    # (N, C, kernel_size_flat, patches)
        pooled = self.pool(patches)                                                               # (N, C, patches)
        Y = pooled.reshape(N, C, H_out, W_out)                                                    # (N, C, H_out, W_out)

        return Y  # (N, C, H_out, W_out)

    def pool(self, patches):
        raise Exception('Not implemented')


class MaxPool2d(Pool2d):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__(kernel_size, stride, padding, dilation, padding_fill_value=-torch.inf)  # use padding with inf negatives for correct max pooling of negative inputs

    def pool(self, patches):
        max_pooled, _ = patches.max(dim=2)  # (N, C, k*k, patches) -> (N, C, patches)
        return max_pooled

    def __repr__(self):
        return f'MaxPool2d({self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'


class AvgPool2d(Pool2d):
    def pool(self, patches):
        return patches.mean(dim=2)  # (N, C, k*k, patches) -> (N, C, patches)

    def __repr__(self):
        return f'AvgPool2d({self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})'


class BatchAddPool(Module):  # adds features across batch dimension with batch_index provided (i.e. scatter_add)

    def forward(self, x, batch_index=None):
        assert len(x.shape) == 2, f'Expected 2d tensor, got {x.shape}'
        if batch_index is not None:
            batch_index = batch_index.view(-1, 1).expand(x.shape)
            batch_size, dim_h = batch_index.max().item() + 1, x.shape[-1]

            x_sums = torch.zeros((batch_size, dim_h), device=x.device, dtype=x.dtype)
            x_sums.scatter_add_(dim=0, index=batch_index, src=x)
        else:
            x_sums = x.sum(dim=0, keepdim=True)

        return x_sums

class SEGate(Module):
    """
    Paper: Squeeze-and-Excitation Gate layer
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, channels, reduction=16):
        self.weight  = Param((2, channels, channels//reduction))  # concatenated weights of the two linear layers
        self.channels = channels
        self.reduction = reduction
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.kaiming_normal_relu_(self.weight, self.channels)

    def forward(self, x):
        assert len(x.shape) == 4, f'Expected 4D input (N, C, H, W), but got {x.shape}'
        N, C, H, W = x.shape

        # Squeeze (channel-wise) - provides global (spatially unrestricted) information
        z = x.mean(dim=(2, 3))                         # (N, C, H, W) -> (N, C)

        # Excitation gate (adaptive recalibration)
        z = relu(z @ self.weight[0])                   # (N, C) -> (N, R)
        z = z @ self.weight[1].T                       # (N, R) -> (N, C)
        p = sigmoid(z)  # sigmoid ensures the probs aren't mutually-exclusive

        # Scale input features (self-attention)
        x = x * p.view(N, C, 1, 1)                     # (N, C, H, W)
        return x

    def __repr__(self):
        return f'SEGate({self.channels}, {self.reduction}): {self.n_params} params'


class Graph_cell(Module):  # graph layer with node-wise neighborhood function
    def __init__(self, in_channels, out_channels):
        self.weight_neighbors = Param((in_channels, out_channels))  # (c_in, c_out)
        self.weight_self = Param((in_channels, out_channels))        # (c_in, c_out)
        self.bias = Param((1, out_channels))
        self.in_channels, self.out_channels = in_channels, out_channels
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.kaiming_normal_relu_(self.weight_self, self.in_channels)
        init.kaiming_normal_relu_(self.weight_neighbors, self.in_channels)
        self.bias.zero_()

    def forward(self, X, A):
        b, n, c = X.shape
        assert A.shape == (b, n, n)

        deg = A.sum(dim=1).to_dense().view(b, n, 1)
        message = A @ X * torch.where(deg != 0, 1 / deg, 0)                        # message function m = (node:opt, neighbors, edges)
        X = message @ self.weight_neighbors + X @ self.weight_self + self.bias     # update function  h = (node, m)
        return X


class GCN_cell(Module):
    """
    Paper: Semi-Supervised Classification with Graph Convolutional Networks
    https://arxiv.org/pdf/1609.02907.pdf
    """
    def __init__(self, in_channels, out_channels):
        self.weight = Param((in_channels, out_channels))  # (c_in, c_out) - shared for self and neighbors connections
        self.bias = Param((1, out_channels))
        self.in_channels, self.out_channels = in_channels, out_channels
        self.reset_parameters()
        self._cache = {'adjacency': None, 'adjacency_normalized': None}

    @torch.no_grad()
    def reset_parameters(self):
        init.kaiming_normal_relu_(self.weight, self.in_channels)
        self.bias.zero_()

    def forward(self, X, A):
        b, n, c = X.shape
        assert A.shape == (b, n, n)

        A_norm = self._normalize_adjacency(A)
        X = A_norm @ X @ self.weight + self.bias
        return X

    def _normalize_adjacency(self, A):
        if self._cache['adjacency'] is A:
            return self._cache['adjacency_normalized']

        b, n, n = A.shape

        I = identity(n, sparse=False, device=A.device)
        A_self = A + I                                   # add self connections
        D = I * A_self.sum(dim=1, keepdims=True)         # diagonal degree matrix
        D[D != 0] = D[D != 0] ** (-1 / 2)                # inverse squared degree matrix
        A_norm = D @ A_self @ D                          # normalized adjacency matrix

        self._cache['adjacency'] = A
        self._cache['adjacency_normalized'] = A_norm
        return A_norm


class GraphSAGE_cell(Module):  # SAGE = SAmple and aggreGatE
    """
    Paper: Inductive Representation Learning on Large Graphs
    https://arxiv.org/pdf/1706.02216.pdf
    """
    def __init__(self, in_channels, hidden_channels, aggregation='maxpool'):
        assert aggregation in ('neighbor', 'maxpool', 'meanpool', 'mean'), f'Invalid aggregation operator: {aggregation}'

        self.weight_self = Param((in_channels, hidden_channels))       # self (c_in, c_out)
        self.weight_neighbors = Param((in_channels, hidden_channels))  # neighbors (c_in, c_out)
        self.bias_self = Param((1, hidden_channels))
        self.bias_neighbors = Param((1, hidden_channels))
        if aggregation in ('maxpool', 'meanpool'):
            self.weight_pool = Param((in_channels, in_channels))
            self.bias_pool = Param((1, in_channels))

        self.in_channels, self.hidden_channels = in_channels, hidden_channels
        self.out_channels = self.hidden_channels * 2
        self.aggregate_operator = aggregation
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for name, param in self.parameters():
            if 'weight' in name:
                init.kaiming_normal_relu_(param, self.in_channels)
            else:  # bias
                param.zero_()

    def forward(self, X, A):
        b, n, c = X.shape
        assert A.shape == (b, n, n)

        message = self.aggregate(X, A)
        X = torch.cat((  # project without mixing self to neighbors features
            message @ self.weight_neighbors + self.bias_self,
            X @ self.weight_self + self.bias_neighbors  # can be viewed as dense (skip) connection
        ), dim=-1)

        # Skip activation and normalization, to use a modular approach
        # X = relu(X)
        # X /= X.norm(dim=-1, keepdim=True)

        return X  # (b, n, c) -> (b, n, 2c)

    def aggregate(self, X, A):
        b, n, c = X.shape
        if self.aggregate_operator == 'neighbor':
            deg = A.sum(dim=1).to_dense().view(1, n, 1)
            message = A @ X * torch.where(deg != 0, 1 / deg, 0)     # A @ X / deg  == self.get_neighbor_features(X, A).sum(dim=2) / deg

        elif self.aggregate_operator == 'mean':
            message = A @ X / n                                     # A @ X / n == self.get_neighbor_features(X, A).mean(dim=2)

        elif self.aggregate_operator == 'meanpool':
            H = relu(X @ self.weight_pool + self.bias_pool)         # trainable projection
            message = A @ H / n                                     # A @ H / n == self.get_neighbor_features(H, A).mean(dim=2)

        elif self.aggregate_operator == 'maxpool':  # (paper) samples fixed number of neighbors
            H = relu(X @ self.weight_pool + self.bias_pool)         # trainable projection
            neighbor_features = self.get_neighbor_features(H, A)    # for each node, collect all adjacent node features with broadcasting (N, N, c)
            message, _ = neighbor_features.max(dim=2)               # and then select only the max features values across each adjacent nodes

        else:
            raise ValueError
        return message

    def get_neighbor_features(self, X, A):  # neighbor sampling is done on data level (as mini-batched subgraphs)
        b, n, c = X.shape
        neighbor_features = X.view(b, 1, n, c) * A.to_dense().view(b, n, n, 1)     # for each node, collect all adjacent node features with broadcasting (N, N, c)
        return neighbor_features


class DiffPool(Module):
    """
    Paper: Hierarchical Graph Representation Learning with Differentiable Pooling
    https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf
    """
    def forward(self, Z, A, S_logit):  # S is the cluster assignment matrix
        b, n, c = Z.shape
        assert A.shape == (b, n, n) and S_logit.shape[:2] == (b, n)

        S = softmax(S_logit, dim=1)
        ST = S.permute(0, 2, 1)

        X = ST @ Z            # (b, m, n) @ (b, n, c) -> (b, m, c)
        A_new = ST @ A @ S    # (b, m, n) @ (b, n, n) @ (b, n, m) -> (b, m, m)

        # Compute additional losses
        loss_link = torch.norm(A - S @ ST)  / A.numel()   # to pool nearby nodes together
        loss_entropy = entropy(S_logit, logits=True)      # to have S vectors close to one-hot

        return X, A_new, (loss_link, loss_entropy)


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class GELU(Module):
    """
    Paper: Gaussian Error Linear Units (GELUs)
    https://arxiv.org/pdf/1606.08415v5
    """
    def __init__(self, approx_tanh=False):
        self.approx_tanh = approx_tanh

    def forward(self, x):
        if self.approx_tanh:
            return gelu_tanh_approx(x)
        return gelu(x)


class GLU(Module):  # Gated Linear Unit
    """
    Paper: Language Modeling with Gated Convolutional Networks
    https://arxiv.org/pdf/1612.08083v3
    """

    def __init__(self, input_size, hidden_size, bias=True, gate_fn=sigmoid):
        assert hidden_size % 2 == 0, f'hidden_size must be an even, but got {hidden_size}'
        self.weight = Param((input_size, hidden_size))
        self.bias = Param((1, hidden_size)) if bias else 0.
        self.gate = gate_fn
        self.input_size, self.output_size = input_size, hidden_size
        self.has_bias = bias
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.linear_uniform_(self.weight, self.input_size)
        if self.has_bias:
            init.linear_uniform_(self.bias, self.input_size)

    def forward(self, x):
        # Concatenated equivalent to: y = (x @ W + b) * self.gate(x @ V + c)
        a, b = (x @ self.weight + self.bias).chunk(2, dim=-1)
        y = a * self.gate(b)
        return y

    def __repr__(self):
        return f'{self.__class__.__name__}({self.input_size}, {self.output_size}, bias={self.has_bias}): {self.n_params} params'



class SwiGLU(GLU):  # Swish-Gated Linear Unit
    """
    Paper: GLU Variants Improve Transformer
    https://arxiv.org/pdf/2002.05202
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, gate_fn=swish)  # swish(beta=1) is the same as silu()



class Flatten(Module):
    def __init__(self, start_dim=1):
        self.start_dim = start_dim

    def forward(self, x):
        return x.flatten(self.start_dim)


class DotProductAttention(Module):
    def __init__(self, scaled=True, dropout=0.):
        self.scaled = scaled
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, query, key, value, attn_mask=None):
        (b, q, emb), (b, k, emb), (b, k, emb_v) = query.shape, key.shape, value.shape
        assert emb == query.shape[-1] == key.shape[-1], f'Expected same vector length for query {query.shape} and key {key.shape}'
        assert k == key.shape[1] == value.shape[1], f'Expected same number of key-values pairs of key {query.shape} and value {value.shape}'

        Q, K_T, V = query, key.permute(0, 2, 1), value
        scale = sqrt(emb) if self.scaled else 1.

        # Compute attention weights
        z = Q @ K_T / scale                      # (b, q, k)  <- (b, q, emb) @ (b, emb, k)
        a = softmax(z, dim=-1, ignore_mask=attn_mask)
        if self.dropout:
            a = self.dropout.forward(a)

        # Weighted sum of values for each query
        v = a @ V                                # (b, q, emb_v)  <- (b, q, v) @ (b, k, emb_v)
        return v, a


class AdditiveAttention(Module):
    """
    Paper: Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/pdf/1409.0473.pdf
    - called Alignment model
    """
    def __init__(self, query_size, key_size, hidden_size, dropout=0.):
        self.weight_query = Param((query_size, hidden_size))  # (emb_q, h)
        self.weight_key   = Param((key_size, hidden_size))    # (emb_k, h)
        self.weight_value = Param((hidden_size,))                    # (h)
        self.dropout = Dropout(dropout)

        self.query_size, self.key_size, self.hidden_size = query_size, key_size, hidden_size
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.linear_uniform_(self.weight_query, in_size=self.query_size)
        init.linear_uniform_(self.weight_key, in_size=self.key_size)
        init.linear_uniform_(self.weight_value, in_size=self.hidden_size)

    def forward(self, query, key, value, attn_mask=None):
        (b, q, emb_q), (b, k, emb_k), (b, k, emb_v), h = query.shape, key.shape, value.shape, self.hidden_size
        assert k == key.shape[1] == value.shape[1], f'Expected same number of key-values pairs of key {query.shape} and value {value.shape}'

        # Compute additive scores
        Hq = query @ self.weight_query                  # (b, q, h)
        Hk = key @ self.weight_key                      # (b, k, h)  <- can be precomputed for each t, because it doesn't depend on the query
        H = tanh(Hq.unsqueeze(2) + Hk.unsqueeze(1))     # (b, q, k, h)  <- broadcasted sum to support multiple queries
        z = H @ self.weight_value                       # (b, q, k)  <- (b, q, k, h) @ (h)   scores of each key for each query

        # Compute attention weights
        a = softmax(z, dim=-1, ignore_mask=attn_mask)
        if self.dropout:
            a = self.dropout.forward(a)

        # Weighted sum of values for each query
        out = a @ value                                 # (b, q, emb_v)  <- (b, q, v) @ (b, k, emb_v)
        return out, a


class DiagBlockAttention(Module):
    """
    Paper: Generating Long Sequences with Sparse Transformers
    https://arxiv.org/pdf/1904.10509
    """

    def __init__(self, block_size=4, dropout=0.):
        self.block_size = block_size  # stride
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, Q, K, V, is_causal=True):  # local diagonal blocks
        b, t, e = Q.shape
        assert is_causal is True, 'Only causal masking is supported'
        assert Q.shape == K.shape == V.shape, f'Expecting same shape of Q{Q.shape}, K{K.shape}, V{V.shape}'
        assert t % self.block_size == 0, f'The sequence length {t} is not divisible by the block size {self.block_size}'
        K_T = K.mT
        block_size = self.block_size
        n_blocks = t // block_size

        # Compute block-diagonal of the attention pattern (local attention)
        Q_blocks = Q.view(b, n_blocks, block_size, e)  # (b, n_blocks, block_size, e)
        K_T_blocks = K_T.view(b, e, n_blocks, block_size).permute(0, 2, 1, 3)  # (b, n_blocks, e, block_size)
        Z_diag_blocks = Q_blocks @ K_T_blocks / sqrt(e)
        Z_diag_blocks = Z_diag_blocks.view(b, t, block_size)

        # Apply causal mask
        causal_mask = self.get_causal_mask(t).to(Z_diag_blocks.device)
        Z_diag_blocks = Z_diag_blocks.masked_fill(causal_mask, -torch.inf)

        # Softmax scores
        A = softmax(Z_diag_blocks, dim=-1)
        if self.dropout:
            A = self.dropout.forward(A)

        # Weighted values
        out = A.view(b, t//block_size, block_size, block_size) @ V.view(b, t // block_size, block_size, e)  # (b, n_blocks, block_size, e)
        out = out.view(b, t, e)

        return out, A

    def expand_attn_blocks(self, A_blocks):
        b, t, _ = A_blocks.shape
        A = torch.zeros(b, t, t, device=A_blocks.device)
        causal = self.get_causal_mask(t)
        attn_mask = self.get_attn_mask(t)
        A[:, ~attn_mask] = A_blocks[:, ~causal]
        return A

    def get_causal_mask(self, t):
        block_size = self.block_size
        causal = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        causal = causal.repeat(t//block_size + 1, 1)[:t]
        return causal

    def get_attn_mask(self, t):
        mask = torch.zeros(t, t)
        for i in range(0, t, self.block_size):  # local block diagonal
            mask[i:i + self.block_size, i:i + self.block_size] = 1
        mask = torch.tril(mask)  # causal
        mask = ~mask.bool()
        return mask


class ColumnBlockAttention(Module):
    """
    Paper: Generating Long Sequences with Sparse Transformers
    https://arxiv.org/pdf/1904.10509
    """

    def __init__(self, block_size=4, dropout=0.):
        self.block_size = block_size  # stride
        self.dropout = Dropout(dropout) if dropout > 0 else None


    def forward(self, Q, K, V, is_causal=True):  # global columns strided by block_size
        b, t, e = Q.shape
        assert is_causal is True, 'Only causal masking is supported'
        assert Q.shape == K.shape == V.shape, f'Expecting same shape of Q{Q.shape}, K{K.shape}, V{V.shape}'
        assert t % self.block_size == 0, f'The sequence length {t} is not divisible by the block size {self.block_size}'
        K_T = K.mT
        block_size = self.block_size
        cols = torch.arange(block_size-1, t, block_size)

        # Compute strided columns of the attention pattern (global attention)
        Z_cols = Q @ K_T[:, :, cols] / sqrt(e)

        # Apply causal mask
        causal_mask = self.get_causal_mask(t).to(Z_cols.device)
        Z_cols = Z_cols.masked_fill(causal_mask, -torch.inf)

        # Softmax scores
        A = torch.zeros_like(Z_cols)        # (b, t, cols)
        A[:, block_size-1:] = softmax(Z_cols[:, block_size-1:], dim=-1)   # skips initial timestamps where there are no columns (so all values are -inf)
        if self.dropout:
            A = self.dropout.forward(A)

        # Weighted values
        out = A @ V[:, (rows := cols)]  # (b, t, cols) @ (b, t[rows], e) -> (b, t, e)

        return out, A

    def expand_attn_blocks(self, A_blocks):
        b, t, _ = A_blocks.shape
        A = torch.zeros(b, t, t, device=A_blocks.device, dtype=A_blocks.dtype)
        causal = self.get_causal_mask(t)
        attn_mask = self.get_attn_mask(t)
        A[:, ~attn_mask] = A_blocks[:, ~causal]
        return A

    def get_causal_mask(self, t):
        block_size = self.block_size
        if self.block_size <= t:
            cols = torch.arange(block_size - 1, t, block_size)
            causal = torch.arange(t).view(t, 1).expand(t, len(cols)) < cols
        else:
            causal = torch.tensor([], dtype=torch.bool).view(t, 0)
        return causal

    def get_attn_mask(self, t):
        mask = torch.zeros(t, t)
        if self.block_size <= t:  # global strided columns
            cols = torch.arange(self.block_size - 1, t, self.block_size)
            mask[:, cols] = 1
        mask = torch.tril(mask)  # causal
        mask = ~mask.bool()
        return mask


class MultiHeadAttention(Module):
    """
    Paper: Attention Is All You Need
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, embed_dim, n_heads, dropout=0., n_kv_heads=None, bias=False):
        assert embed_dim % n_heads == 0, f'input_size {embed_dim} must be divisible by n_heads {n_heads}'
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads  # used only by GroupedQueryAttention
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        self.has_bias = bias
        self.attn_weights = None  # keep record of the last attention weights for visualization

        self.weight_q = Param((embed_dim, self.head_dim * self.n_heads))     # not concatenated to weight_qkv for flexibility (while extending the class)
        self.weight_k = Param((embed_dim, self.head_dim * self.n_kv_heads))  # but the price is performance
        self.weight_v = Param((embed_dim, self.head_dim * self.n_kv_heads))  #
        self.weight_o = Param((embed_dim, embed_dim))

        self.bias_q = Param((1, self.head_dim * self.n_heads)) if bias else 0.
        self.bias_k = Param((1, self.head_dim * self.n_kv_heads)) if bias else 0.
        self.bias_v = Param((1, self.head_dim * self.n_kv_heads)) if bias else 0.
        self.bias_o = Param((1, embed_dim)) if bias else 0.

        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.xavier_uniform_(self.weight_q, *self.weight_q.shape)
        init.xavier_uniform_(self.weight_k, *self.weight_k.shape)
        init.xavier_uniform_(self.weight_v, *self.weight_v.shape)
        init.xavier_uniform_(self.weight_o, *self.weight_o.shape)
        if self.has_bias:
            self.bias_q.data.zero_()
            self.bias_k.data.zero_()
            self.bias_v.data.zero_()
            self.bias_o.data.zero_()


    def forward(self, query, key, value, attn_mask=None, flash=False, transform=None, kv_cache=None, is_causal=False):
        (b, t_, emb), (b, t, k_dim), (b, t, v_dim) = query.shape, key.shape, value.shape
        assert query.shape[0] == key.shape[0] == value.shape[0], f'Expected same batch_size but got {query.shape=}, {key.shape=}, {value.shape=}'
        assert (b, t) == key.shape[:-1] == value.shape[:-1], f'Expected same number of key-values pairs of key {query.shape} and value {value.shape}'
        attn_mask = self._adapt_attn_mask((b, self.n_heads, t_, t), attn_mask, flash, is_causal, query.device)

        # Project to smaller vectors
        Q = query @ self.weight_q + self.bias_q            # (b, t', emb)   <- (b, t', emb) @ (emb, emb)
        K = key @ self.weight_k + self.bias_k              # (b, t,  emb)   <- (b, t,  emb) @ (emb, emb)
        V = value @ self.weight_v + self.bias_v            # (b, t,  emb)   <- (b, k,  emb) @ (emb, emb)

        # Separate the heads
        Q = Q.view(b, t_, self.n_heads, self.head_dim)     # (b, t', h, head_emb)
        K = K.view(b, t, self.n_kv_heads, self.head_dim)   # (b, t,  h, head_emb)
        V = V.view(b, t, self.n_kv_heads, self.head_dim)   # (b, t,  h, head_emb)

        # Apply custom transformation on the projections before heads splitting and dot attention (e.g. rotary encoding, KV cache)
        if transform is not None:
            Q, K, V = transform(Q, K, V)

        # Split projections into n_heads
        Q = self._split_heads(Q)                          # (b, h, t', head_dim)
        K = self._split_heads(K)                          # (b, h, t,  head_dim)
        V = self._split_heads(V)                          # (b, h, t,  head_dim)

        # Compute attention (heads are considered batches)
        out, self.attn_weights = self.dot_product_attention(Q, K, V, attn_mask, is_causal, flash)
        out = self._merge_heads(out)                      # (b, t', emb)   <- concat the heads to emb

        # Finally project the weighted values
        out = out @ self.weight_o + self.bias_o           # (b, t', emb)   <- (b, t', emb) @ (emb, emb)

        return out

    def dot_product_attention(self, Q, K, V, attn_mask=None, is_causal=False, flash=False):
        if flash:
            # To ensure FlashAttention backend is used wrap in: with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if attn_mask is not None:
                warnings.warn("FlashAttention2 doesn't support custom attn_mask, you should use is_causal=True")  # however the other optimized backends will work with: attn_mask=~attn_mask
                attn_mask = ~attn_mask  # adapt to the API
            self.attn_weights = None
            return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, attn_mask=attn_mask), None

        Z = Q @ K.mT / sqrt(self.head_dim)                # (b, h, t', t)  <- (b, h, t', head_dim)  @  (b, h, head_dim, t)
        A = softmax(Z, dim=-1, ignore_mask=attn_mask)
        if self.dropout:
            A = self.dropout.forward(A)
        out = A @ V                                       # (b, h, t', head_dim)  <- (b, h, t', t) @ (b, h, t, head_dim)
        return out, A

    def _split_heads(self, X):
        return ein.rearrange(X, 'b t h d -> b h t d', h=self.n_heads)

    def _merge_heads(self, X):
        return ein.rearrange(X, 'b h t d -> b t (h d)', h=self.n_heads)

    def _adapt_attn_mask(self, shape, attn_mask=None, flash=False, is_causal=False, device=None):  # handle special cases and do some validations
        (b, h, t_, t) = shape

        if is_causal and attn_mask is not None:
            raise Exception('Expected either attn_mask or is_causal, but got them both')

        elif attn_mask is not None:
            if attn_mask.shape == (t_, t):
                attn_mask = attn_mask.view(1, 1, t_, t)
            elif attn_mask.ndim == 3 and attn_mask.shape[0] == b * h:
                attn_mask = attn_mask.view(b, h, *attn_mask.shape[1:])
            elif attn_mask.ndim == 4:
                pass  # good
            else:
                f'Expected attn_mask shape {t_, t} or {b * h, t_, t}, but got {attn_mask.shape}'

        elif is_causal and not flash and t_ > 1:  # for flash attention, we don't need to create an attention mask
            attn_mask = torch.triu(torch.ones(t_, t), diagonal=1).bool().view(1, 1, t_, t).to(device)  # can be cached

        return attn_mask

    def get_last_attn_weights(self):
        return self.attn_weights

    def __repr__(self):
        return f'{self.__class__.__name__}({self.embed_dim}, n_heads={self.n_heads}, bias={self.has_bias}): {self.n_params} params'



class GroupedQueryAttention(MultiHeadAttention):
    """
    Paper: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
    https://arxiv.org/pdf/2305.13245v3
    """

    def __init__(self, embed_dim, n_heads, groups, dropout=0., bias=False):
        assert n_heads % groups == 0, f'n_heads {n_heads} must be divisible by groups {groups}'
        self.groups = groups
        super(GroupedQueryAttention, self).__init__(embed_dim, n_heads, dropout, n_kv_heads=groups, bias=bias)

    def _split_heads(self, X):
        is_shared_kv = self.n_kv_heads == X.shape[2] != self.n_heads
        return ein.repeat(X, 'b t h d -> b (h repeat) t d', d=self.head_dim, repeat=self.n_heads // self.n_kv_heads if is_shared_kv else 1)

    def __repr__(self):
        return f'GroupedQueryAttention({self.embed_dim}, n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, groups={self.groups}): {self.n_params} params'



class RelativeWindowAttention(MultiHeadAttention):

    def __init__(self, embed_dim, n_heads, window_size, dropout=0., bias=False):
        self.window_size = window_size
        self.n_patches = window_size * window_size
        self.relative_pos = RelativePositionBias2d(window_size, window_size, n_heads)
        super(RelativeWindowAttention, self).__init__(embed_dim, n_heads, dropout, n_kv_heads=None, bias=bias)

    def dot_product_attention(self, Q, K, V, attn_mask=None, is_causal=False, flash=False):
        assert not flash, f'Flash Attention is not implemented'
        assert self.n_patches == Q.shape[-2] == K.shape[-2]
        B = self.relative_pos.get_bias_grid()

        # Attention with relative position bias
        Z = Q @ K.mT / sqrt(self.head_dim) + B            # (b, h, t', t)  <- (b, h, t', head_dim)  @  (b, h, head_dim, t) + (t', t)
        A = softmax(Z, dim=-1, ignore_mask=attn_mask)
        if self.dropout:
            A = self.dropout.forward(A)
        out = A @ V                                       # (b, h, t', head_dim)  <- (b, h, t', t) @ (b, h, t, head_dim)
        return out, A


class SparseMultiHeadAttention(MultiHeadAttention):
    """
    Paper: Generating Long Sequences with Sparse Transformers
    https://arxiv.org/pdf/1904.10509
    """

    def __init__(self, embed_dim, n_heads, dropout=0., block_size=4, bias=False):
        super(SparseMultiHeadAttention, self).__init__(embed_dim, n_heads, dropout=dropout, bias=bias)
        self.block_size = block_size  # stride
        self.attn_local = DiagBlockAttention(block_size, dropout)
        self.attn_global = ColumnBlockAttention(block_size, dropout)

    def dot_product_attention(self, Q, K, V, attn_mask=None, is_causal=True, flash=False):
        assert is_causal is True, 'Only causal masking is supported'
        assert attn_mask is None, f'Custom attention mask is not supported, will be applying just the causal mask'
        assert not flash, f'Flash attention is not supported for SparseMultiHeadAttention'
        assert Q.shape == K.shape == V.shape, f'Expecting same shape of Q{Q.shape}, K{K.shape}, V{V.shape}'
        b, t, e = Q.shape

        # used for inference to match the attention patterns blocks. For example for x[:, t=1] will be temporary padded to x[:, block-size] to match to the attention patterns
        is_block_padded = t % self.block_size != 0
        if is_block_padded:
            pad = self.block_size - t % self.block_size
            Q, K, V = [torch.cat((x, torch.zeros(b, pad, e, device=x.device)), dim=1) for x in (Q, K, V)]

        # Compute the values and attention
        out_local, A_local = self.attn_local(Q, K, V)
        out_global, A_global = self.attn_global(Q, K, V)
        out = out_local + out_global  # note: that's not the same to the union of column and diag blocks patterns, as here the output is the sum of both patterns computed independently

        if is_block_padded:
            out = out[:, :t]
            A_local, A_global = A_local[:, :t], A_global[:, :t, :t//self.block_size]

        return out, (A_local, A_global)

    def _split_heads(self, X):
        return ein.rearrange(X, 'b t h d -> (b h) t d', h=self.n_heads)

    def _merge_heads(self, X):
        return ein.rearrange(X, '(b h) t d -> b t (h d)', h=self.n_heads)

    def _adapt_attn_mask(self, shape, attn_mask=None, flash=False, is_causal=False, device=None):
        assert is_causal is True, 'Only causal masking is supported'
        assert attn_mask is None, f'Custom attention mask is not supported, will be applying just the causal mask'
        return None


    @torch.no_grad()
    def get_last_attn_weights(self):
        A_local, A_global = self.attn_weights

        # Generate dense attention matrix from the blocks ( it can be made as sparse matrix if necessary)
        A = self.attn_local.expand_attn_blocks(A_local) + self.attn_global.expand_attn_blocks(A_global)  # importantly we sum the overlaps between local and global patterns

        return ein.rearrange(A, '(b h) tt ts -> b h tt ts', h=self.n_heads)



class KVCache:
    def __init__(self):
        self.cache_kv = None

    def reset(self):
        self.cache_kv = None

    def update(self, k, v, pos, reset=False):
        assert torch.no_grad(), f'Expecting a key-value cache to be used only during inference'
        assert pos == 0 or (v.shape[1] == v.shape[1] == 1), f'Expected kv_cache to be used for inference of the next token, but detected batched usage. Investigate!'

        if reset:
            self.reset()

        if not self.cache_kv:
            self.cache_kv = k, v
        else:
            self.cache_kv = (
                torch.cat((self.cache_kv[0], k), dim=1),
                torch.cat((self.cache_kv[1], v), dim=1),
            )
        k, v = self.cache_kv
        return k, v



class PositionalEncoding(Module):
    """
    Paper: Attention Is All You Need
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, depth_size, max_seq_len, dropout=0., mixed=False, base_freq_theta=10_000):
        encodings = self.compute_encodings(depth_size, max_seq_len, mixed, base_freq_theta)
        self.fixed_embeddings = self.register_buffer('fixed_embeddings', encodings)
        self.dropout = Dropout(dropout) if dropout else None
        self.depth_size, self.max_seq_len = depth_size, max_seq_len

    @staticmethod
    def compute_encodings(depth_size, seq_len, mixed=False, base_freq_theta=10_000):
        assert depth_size % 2 == 0, f'depth_size must be even, got {depth_size}'

        pos = torch.arange(seq_len)
        i = torch.arange(depth_size//2)
        freq = 1 / base_freq_theta ** (2 * i / depth_size)
        radians = pos.view(-1, 1) * freq                  # (t, d/2)  <-  (t, 1) * (d/2)
        sin, cos = torch.sin(radians), torch.cos(radians)

        if not mixed:  # Concatenation is equivalent to interleaving (i.e. mixed=True), because it is just a permutation of independent channels
            embeddings = torch.cat((sin, cos), dim=-1)
        else:
            embeddings = torch.zeros(seq_len, depth_size)
            embeddings[:, 0::2] = cos   # switched cos and sin to match to rotary freqs
            embeddings[:, 1::2] = sin

        return embeddings   # (t, d)

    def forward(self, x):
        B, T, E = x.shape
        assert E == self.depth_size, f'Expected embedding size of {self.depth_size} but got {x.shape}'
        assert T <= self.max_seq_len, f'Too long sequence {x.shape}, expected max length size to be {self.max_seq_len}'  # it can be lazy extended

        x = x + self.fixed_embeddings[:T].view(1, T, E).to(x.device)
        if self.dropout:
            x = self.dropout.forward(x)
        return x

    def plot(self):
        plt.pcolormesh(self.fixed_embeddings.T)
        plt.title('Sinusoidal embeddings')
        plt.xlabel('Position')
        plt.ylabel('Depth')
        plt.colorbar()
        plt.show()

    def __repr__(self):
        return f'PositionalEncoding(depth_size={self.depth_size}, max_seq_len={self.max_seq_len}, dropout={self.dropout.p if self.dropout is not None else None}): {self.fixed_embeddings.shape}'


class RotaryEncoding(Module):
    """
    Paper: RoFormer: Enhanced Transformer with Rotary Position Embedding
    https://arxiv.org/pdf/2104.09864
    """

    def __init__(self, emb_dim, max_seq_len, base_freq_theta=10_000):
        self.fixed_embeddings = self.compute_encodings(emb_dim, max_seq_len, base_freq_theta)
        self.emb_dim, self.max_seq_len = emb_dim, max_seq_len

    @staticmethod
    def compute_encodings(d, seq_len, base_freq_theta=10_000):
        assert d % 2 == 0, f'embedding dim must be even, got {d}'
        pos = torch.arange(seq_len).view(-1, 1)
        i = torch.arange(d // 2)
        theta = 1 / base_freq_theta ** (2 * i / d)    # different frequency for each position
        freqs_complex = torch.exp(pos * theta * 1j)   # equivalent to cis: cos(x) + isin(x) = e^{ix}
        return freqs_complex   # (t, d/2)

    def forward(self, x, pos=0, clockwise=False):
        if x.dtype is torch.bfloat16:  # complex numbers don't support bfloat16 (which is used in mixed-precision / autocasting)
            return self.forward(x.float(), pos, clockwise).bfloat16()

        b, t, h, d = x.shape
        rotation = self.fixed_embeddings.to(x.device)
        if clockwise:
            rotation = rotation.conj()

        x = torch.view_as_complex(x.view(b, t, h, -1, 2))   # split in pairs of dims, and represent each two real numbers as complex number
        x = x * rotation[pos:pos+t].view(1, t, 1, d//2)     # rotate on the complex plane: x * e^{pos * theta * 1j}
        x = torch.view_as_real(x).view(b, t, h, d)          # restore back to real numbers  (b, t, h, id//2) -> (b, t, h, d)
        return x

    def plot(self):
        t, d = self.fixed_embeddings.shape[0], self.emb_dim
        emb = torch.view_as_real(self.fixed_embeddings).view(t, d)
        plt.pcolormesh(emb.T)
        plt.title('Rotary sinusoidal embeddings')
        plt.xlabel('Position')
        plt.ylabel('Depth')
        plt.colorbar()
        plt.show()

        with torch.no_grad():
            x = torch.ones(1, t, d)
            plots.rotary_encoded_vectors(x, self.forward(x), max_plots=6)


class PatchEmbedding(Module):
    """
    Paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://arxiv.org/pdf/2010.11929
    """

    def __init__(self, patch_size, embed_size, in_channels, keep_img_dim=False, bias=True):  # todo: flip order of embed_size / in_channels
        self.patch_size, self.embed_size, self.in_channels = patch_size, embed_size, in_channels
        self.n_pixels = in_channels * patch_size * patch_size  # sequence of pixels for each patch

        # Note: Convolution with stride=kernel_size=patch_size is the same as splitting into patches and projecting linearly
        self.proj = Conv2d(in_channels, out_channels=embed_size, kernel_size=patch_size, stride=patch_size, padding=0, bias=bias)
        self.keep_dim = keep_img_dim

    def forward(self, X):
        (B, C, H, W), P = X.shape, self.patch_size
        assert W == H and W % P == 0
        T = (H // P) * (W // P)

        # Project and flatten to embed size:
        x = self.proj(X)                     # (B, C, H, W) -> (B, E, H//P, W//P)
        if not self.keep_dim:
            x = x.flatten(start_dim=2).mT    # (B, T, C)
        return x


class RelativePositionBias2d(Module):
    """
    Paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    https://arxiv.org/pdf/2103.14030
    """

    def __init__(self, img_height, img_width, n_heads=1):
        self.height = img_height
        self.width = img_width
        self.max_i = 2 * img_height - 1
        self.max_j = 2 * img_width - 1
        self.relative_pos = Param((n_heads, self.max_i, self.max_j))   # B_hat                                 (n_heads, 2*H-1, 2*W-1)
        self.rel_grid_i, self.rel_grid_j = self.generate_grid()        # B = B_hat[:, rel_grid_i, rel_grid_j]  (n_heads, H*W, H*W)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.relative_pos.normal_(std=.02)

    def generate_grid(self):
        H, W = self.height, self.width
        max_i, max_j = self.max_i, self.max_j

        # generate coords on 2D image
        rows, cols = torch.arange(H), torch.arange(W)
        grid = torch.stack(torch.meshgrid(rows, cols, indexing='ij'), dim=-1)  # H, W, ij

        # flatten and subtract (with broadcasting) to generate QxK grid with relative indices
        rel_grid = grid.view(H*W, 1, 2) - grid.view(1, H*W, 2)            # H*W, H*W, ij

        # shift the relative indices to be only positive (i.e. [-2, -1, 0, 1, 2] -> [0, 1, 2, 3, 4])
        rel_grid_i = rel_grid[..., 0] + max_i//2                          # H*W, H*W
        rel_grid_j = rel_grid[..., 1] + max_j//2                          # H*W, H*W

        assert len(rel_grid_i.unique()) == max_i and len(rel_grid_j.unique()) == max_j
        return rel_grid_i, rel_grid_j

    def get_bias_grid(self):
        return self.relative_pos[:, self.rel_grid_i, self.rel_grid_j]     # (n_heads, H*W, H*W)

    def forward(self, x):
        return x + self.get_bias_grid()
