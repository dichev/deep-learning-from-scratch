import torch
import torch.nn.functional as F
from math import sqrt
from lib.functions import init
from lib.functions.activations import relu, tanh, sigmoid, softmax
from lib.functions.losses import entropy
from lib.base import Param, Module, ModuleList, Sequential
from utils.other import conv2d_calc_out_size, conv2d_pad_string_to_int, identity
from collections import namedtuple

Padding = namedtuple('Padding', ('pad_left', 'pad_right', 'pad_top', 'pad_bottom'))

class Linear(Module):
    def __init__(self, input_size, output_size=1, weights_init=init.linear_uniform_):
        self.weight = Param((input_size, output_size))
        self.bias = Param((1, output_size))
        self.input_size, self.output_size = input_size, output_size
        self.weights_initializer = weights_init
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.weights_initializer(self.weight, self.input_size, self.output_size)
        self.weights_initializer(self.bias, self.input_size, self.output_size)

    def forward(self, X):
        z = X @ self.weight + self.bias    # (N, D)x(D, C) + (1, C)  --> (N, C)
        return z

    def __repr__(self):
        return f'Linear({self.input_size}, {self.output_size}, bias=true): {self.n_params} params'


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

        self.running_mean = torch.zeros(shape)
        self.running_var = torch.ones(shape)
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

    def to(self, device):
        super().to(device)
        self.running_mean.data = self.running_mean.to(device)
        self.running_var.data = self.running_mean.to(device)

    def forward(self, x):
        assert len(x.shape) == len(self.dims) + 1, f'Expect tensor with {len(self.dims) + 1} axis, but got {x.shape}'

        # mini-batch statistics
        if torch.is_grad_enabled():
            mu, var, var_unbias = x.mean(dim=self.dims, keepdims=True), x.var(dim=self.dims, correction=0, keepdims=True), x.var(dim=self.dims, keepdims=True)
            with torch.no_grad():  # fix a memory leak in the computation graph (for var_unbias only)
                self.running_mean = self.decay * self.running_mean + (1 - self.decay) * mu
                self.running_var  = self.decay * self.running_var  + (1 - self.decay) * var_unbias
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
        super().__init__(size, batch_dims=(0, 2, 3))  # note here N, W, H are all considered to be from the batch dim

    def forward(self, x):
        assert len(x.shape) == 4, f'BatchNorm2d layer requires 4D input, but got: {x.shape}'
        N, C, W, H = x.shape
        assert C == self.size, f'Expected {self.size} channels from the input, but got {C}'
        assert max(N, W, H) > 1 or not torch.is_grad_enabled(), 'BatchNorm2d layer requires at least 2 samples'
        return super().forward(x)


class LayerNorm(Module):
    """
    Paper: Layer Normalization
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, size):
        self.shift = Param((1, size))
        self.scale = Param((1, size))
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
    """
    Paper: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
    """

    def __init__(self, p=.5):  # as prob to be zeroed
        assert 0 <= p < 1, f'Dropout probability must be in [0, 1), but got {p}'
        self.p = p

    def forward(self, x):  # note that each sample in the mini-batch is zeroed independently
        if self.p > 0 and torch.is_grad_enabled():
            x = x.clone()
            dropped = torch.rand_like(x) < self.p  # same as torch.bernoulli(x, self.p)
            x[dropped] = 0
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

        self._slice_i = slice(hidden_size * 0, hidden_size * 1)
        self._slice_f = slice(hidden_size * 1, hidden_size * 2)
        self._slice_o = slice(hidden_size * 2, hidden_size * 3)
        self._slice_m = slice(hidden_size * 3, None)

        with torch.no_grad():
            self.bias[:, self._slice_f] = 1.  # set the sigmoid threshold beyond 0.5 to reduce the vanishing gradient at early stages of training (https://proceedings.mlr.press/v37/jozefowicz15.pdf)

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
        h, mem = state
        if h is None:
            h = torch.zeros(N, self.hidden_size, device=x.device)    # fast state
        if mem is None:
            mem = torch.zeros(N, self.hidden_size, device=x.device)  # slow state

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
    def __init__(self, input_size, hidden_size):
        self.weight = Param((input_size + hidden_size, 3 * hidden_size))  # (I+H, 3H)
        self.bias = Param((1, 3 * hidden_size))  # (1, 3H)

        self._slice_r = slice(hidden_size * 0, hidden_size * 1)  # reset gate params
        self._slice_z = slice(hidden_size * 1, hidden_size * 2)  # update gate params
        self._slice_n = slice(hidden_size * 2, hidden_size * 3)  # new cell candidate params

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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
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
        if isinstance(padding, str):  # e.g. valid, same, full
            self.padding = conv2d_pad_string_to_int(padding, kernel_size)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        init.kaiming_normal_relu_(self.weight, self.in_channels * self.kernel_size * self.kernel_size)
        if self.has_bias:
            init.kaiming_normal_relu_(self.bias, 1)

    def forward(self, X):
        N, C, W, H, = X.shape
        W_out, H_out = conv2d_calc_out_size(X, self.kernel_size, self.stride, self.padding, self.dilation)  # useful validation

        """
        # cross-correlation between batch images and filters:
        k = self.kernel_size
        Y = torch.zeros((N, self.out_channels, out_size, out_size), device=X.device)  # (N, C_out, W_out, H_out)
        for h in range(out_size):
            for w in range(out_size):
                for c in range(self.out_channels):
                    patches = X[:, :, w:w+k, h:h+k].reshape(N, -1)
                    kernel = self.weight[c].flatten()
                    bias = self.bias[c]
                    Y[:, c, w, h] = patches @ kernel + bias
        """

        # Vectorized batched convolution: Y = [I] * K + b  (which is actually cross-correlation + bias shifting)
        patches = F.unfold(X, self.kernel_size, self.dilation, self.padding, self.stride)         # (N, kernel_size_flat, patches)
        kernel = self.weight.reshape(self.out_channels, -1)                               # * (channels, kernel_size_flat)
        convolution = torch.einsum('nkp,ck->ncp', patches, kernel)                         # -> (N, channels, patches)
        Y = convolution.reshape(N, self.out_channels, W_out, H_out)                      # (N, channels, out_width, out_height)
        if self.has_bias:
            Y += self.bias.reshape(1, -1, 1, 1)

        return Y  # (N, C_out, W_out, H_out)

    def __repr__(self):
        return f'Conv2d({self.in_channels}, {self.out_channels}, {self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.has_bias}): {self.n_params} parameters'


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
        W_out, H_out = conv2d_calc_out_size(X, self.kernel_size, self.stride, self.padding, self.dilation)  # useful validation

        Y = torch.zeros(N, self.out_channels, W_out, H_out, device=X.device)
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
        N, C, W, H, = X.shape
        if self.kernel_size == W == H and self.stride == 1 and self.dilation == 1 and self.padding == (0, 0, 0, 0):
            # shortcut computations if the pooling is global (but still channel-wise)
            return self.pool(X.view(N, C, -1)).view(N, C, 1, 1)       # (N, C, W, H) -> # (N, C, 1, 1)

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
        assert len(x.shape) == 4, f'Expected 4D input (N, C, W, H), but got {x.shape}'
        N, C, W, H = x.shape

        # Squeeze (channel-wise) - provides global (spatially unrestricted) information
        z = x.mean(dim=(2, 3))                   # (N, C, W, H) -> (N, C)

        # Excitation gate (adaptive recalibration)
        z = relu(z @ self.weight[0])                   # (N, C) -> (N, R)
        z = z @ self.weight[1].T                       # (N, R) -> (N, C)
        p = sigmoid(z)  # sigmoid ensures the probs aren't mutually-exclusive

        # Scale input features (self-attention)
        x = x * p.view(N, C, 1, 1)                     # (N, C, W, H)
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
        a = softmax(z, dim=-1, mask=attn_mask)
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
        a = softmax(z, dim=-1, mask=attn_mask)
        if self.dropout:
            a = self.dropout.forward(a)

        # Weighted sum of values for each query
        out = a @ value                                 # (b, q, emb_v)  <- (b, q, v) @ (b, k, emb_v)
        return out, a



class MultiHeadAttention(Module):
    """
    Attention Is All You Need
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, embed_dim, n_heads, k_dim=None, v_dim=None, scaled=True, dropout=0.):
        assert embed_dim % n_heads == 0, f'input_size {embed_dim} must be divisible by n_heads {n_heads}'
        self.embed_dim = embed_dim
        self.k_dim = embed_dim if k_dim is None else k_dim
        self.v_dim = embed_dim if v_dim is None else v_dim
        self.n_heads = n_heads
        self.scaled = scaled

        self.weight_query = Param((embed_dim, embed_dim))
        self.weight_key   = Param((self.k_dim, embed_dim))
        self.weight_value = Param((self.v_dim, embed_dim))
        self.weight_out   = Param((embed_dim, embed_dim))
        self.dropout = Dropout(dropout) if dropout > 0 else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for name, weight in self.parameters():
            init.xavier_uniform_(weight, *weight.shape)

    def forward(self, query, key, value, attn_mask=None):
        (b, t_, emb), (b, t, k_dim), (b, t, v_dim) = query.shape, key.shape, value.shape
        assert (b, t) == key.shape[:-1] == value.shape[:-1], f'Expected same number of key-values pairs of key {query.shape} and value {value.shape}'

        # Project to smaller vectors
        Q = query @ self.weight_query                                 # (b, t', emb)  <- (b, t',  emb) @ (emb, emb)
        K = key @ self.weight_key                                     # (b, t,  emb)  <- (b, t, k_dim) @ (k_dim, emb)
        V = value @ self.weight_value                                 # (b, t,  emb)  <- (b, k, v_dim) @ (v_dim, emb)

        # Split projections into independent n_heads tensor
        Q   = Q.view(b, t_, self.n_heads, -1).permute(0, 2, 1, 3)     # (b, heads, t', emb/heads)
        K_T = K.view(b, t,  self.n_heads, -1).permute(0, 2, 3, 1)     # (b, heads, emb/heads, t)
        V   = V.view(b, t,  self.n_heads, -1).permute(0, 2, 1, 3)     # (b, heads, t, emb/heads)

        # Compute attention weights for each head
        scale = sqrt(emb//self.n_heads) if self.scaled else 1.
        Z = Q @ K_T / scale                                           # (b, heads, t', t)  <- (b, heads, t', emb/heads)  @  (b, heads, emb/heads, t)
        A = softmax(Z, dim=-1, mask=attn_mask)
        if self.dropout:
            A = self.dropout.forward(A)

        # Weighted sum of values for each query
        out = A @ V                                                   # (b, heads, t', emb/heads)  <- (b, heads, t', t) @ (b, heads, t, emb/heads)
        out = out.permute(0, 2, 1, 3).reshape(b, t_, self.embed_dim)  # (b, t', emb)               <- concat the heads
        out = out @ self.weight_out                                   # (b, t', emb)               <- (b, t', emb) @ (emb, emb)

        return out, A

