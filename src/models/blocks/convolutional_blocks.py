import torch
from lib.layers import Module, ModuleList, Sequential, Conv2d, Conv2dGroups, MaxPool2d, BatchNorm2d, ReLU, SEGate, AvgPool2d, Dropout
from lib.functions.activations import relu


class Inception(Module):
    """
    Paper: Going deeper with convolutions
    https://arxiv.org/pdf/1409.4842.pdf?
    """

    def __init__(self, in_channels, out_channels, spec=(0, (0, 0), (0, 0), 0)):
        c1, (c2_reduce, c2), (c3_reduce, c3), c4 = spec
        assert out_channels == c1 + c2 + c3 + c4, f'Wrong channel spec: expected {out_channels} total output channels, but got {c1}+{c2}+{c3}+{c4}={c1 + c2 + c3 + c4}'

        self.branch1 = Sequential(
            Conv2d(in_channels, c1, kernel_size=1))
        self.branch2 = Sequential(
            Conv2d(in_channels, c2_reduce, kernel_size=1), ReLU(),
            Conv2d(c2_reduce, c2, kernel_size=3, padding='same'))
        self.branch3 = Sequential(
            Conv2d(in_channels, c3_reduce, kernel_size=1), ReLU(),
            Conv2d(c3_reduce, c3, kernel_size=5, padding='same'))
        self.branch4 = Sequential(
            MaxPool2d(kernel_size=3, padding='same'),
            Conv2d(in_channels, c4, kernel_size=1))

    def forward(self, x):
        features = [
            self.branch1.forward(x),
            self.branch2.forward(x),
            self.branch3.forward(x),
            self.branch4.forward(x),
        ]
        x = torch.cat(features, dim=1)  # 4 x (N, C, H, W) ->  (N, 4*C, H, W)
        x = torch.relu(x)  # it might be attached outside the module
        return x


class ResBlock(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, in_channels, out_channels, stride=1, attention=False):
        self.downsampled = stride != 1 or in_channels != out_channels
        self.attention = attention

        self.residual = Sequential(
            Conv2d(in_channels,  out_channels, kernel_size=3, padding='same', stride=stride, bias=False),
            BatchNorm2d(out_channels), ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False),
            BatchNorm2d(out_channels),  # no ReLU
        )
        if self.attention:
            self.se_gate = SEGate(out_channels, reduction=16)
        if self.downsampled:
            self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        y = self.residual.forward(x)
        if self.attention:
            y = self.se_gate.forward(y)
        if self.downsampled:
            x = self.project.forward(x)
        y = relu(y + x)  # add the skip connection
        return y

    def __repr__(self):
        return f'ResBlock(in_channels={self.in_channels}, out_channels={self.out_channels}, stride={self.stride}): {self.n_params} params'


class ResBottleneckBlock(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, in_channels, mid_channels, out_channels, stride=1, attention=False):
        assert mid_channels == out_channels // 4  # just following the models in the paper
        self.downsampled = stride != 1 or in_channels != out_channels
        self.attention = attention

        self.residual = Sequential(
            Conv2d(in_channels,  mid_channels, kernel_size=1, padding='same', stride=stride, bias=False),   # 1x1 conv (reduce channels)
            BatchNorm2d(mid_channels), ReLU(),
            Conv2d(mid_channels, mid_channels, kernel_size=3, padding='same', bias=False),                  # 3x3 conv
            BatchNorm2d(mid_channels), ReLU(),
            Conv2d(mid_channels, out_channels, kernel_size=1, padding='same', bias=False),                  # 1x1 conv (expand channels)
            BatchNorm2d(out_channels),  # no ReLU
        )
        if self.attention:
            self.se_gate = SEGate(out_channels, reduction=16)

        if self.downsampled:
            self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        y = self.residual.forward(x)
        if self.attention:
            y = self.se_gate.forward(y)
        if self.downsampled:
            x = self.project.forward(x)
        y = relu(y + x)  # add the skip connection
        return y

    def __repr__(self):
        return f'ResBottleneckBlock(in_channels={self.in_channels}, mid_channels={self.mid_channels}, out_channels={self.out_channels}, stride={self.stride}): {self.n_params} params'


class ResNeXtBlock(Module):
    """
    Paper: Aggregated Residual Transformations for Deep Neural Networks
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf
    """

    def __init__(self, in_channels, mid_channels, out_channels, groups, stride=1, attention=False):
        assert mid_channels == out_channels // 2  # just following the models in the paper
        self.downsampled = stride != 1 or in_channels != out_channels
        self.attention = attention

        self.residual = Sequential(
            Conv2d(in_channels,  mid_channels, kernel_size=1, padding='same', stride=stride, bias=False),        # 1x1 conv (reduce channels)
            BatchNorm2d(mid_channels), ReLU(),
            Conv2dGroups(mid_channels, mid_channels, kernel_size=3, padding='same', groups=groups, bias=False),  # groups x 3x3 conv with channels/groups
            BatchNorm2d(mid_channels), ReLU(),
            Conv2d(mid_channels, out_channels, kernel_size=1, padding='same', bias=False),                       # 1x1 conv (expand channels)
            BatchNorm2d(out_channels),  # no ReLU
        )
        if self.attention:
            self.se_gate = SEGate(out_channels, reduction=16)
        if self.downsampled:
            self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride

    def forward(self, x):
        y = self.residual.forward(x)
        if self.attention:
            y = self.se_gate.forward(y)
        if self.downsampled:
            x = self.project.forward(x)
        y = relu(y + x)  # add the skip connection
        return y

    def __repr__(self):
        return f'ResNeXtBlock(in_channels={self.in_channels}, mid_channels={self.in_channels}, out_channels={self.out_channels}, groups={self.groups}, stride={self.stride}): {self.n_params} params'


class DenseLayer(Module):
    """
    Paper: Densely Connected Convolutional Networks
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    """

    def __init__(self, in_channels, out_channels, bottleneck_channels_mplr=4, dropout_rate=0., bias=False):
        bottle_channels = out_channels * bottleneck_channels_mplr
        self.layer = Sequential(
            BatchNorm2d(in_channels), ReLU(),  # notice the batch norm is applied before the activation
            Conv2d(in_channels, bottle_channels, kernel_size=1, bias=bias),                   # 1x1 bottleneck
            Dropout(dropout_rate) if dropout_rate else None,
            BatchNorm2d(bottle_channels), ReLU(),
            Conv2d(bottle_channels, out_channels, kernel_size=3, padding='same', bias=bias),  # 3x3 conv
            Dropout(dropout_rate) if dropout_rate else None,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels_mplr = bottleneck_channels_mplr
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.layer.forward(x)
        return x

    def __repr__(self):
        return f'DenseLayer(in_channels={self.in_channels}, out_channels={self.out_channels}, bottleneck_channels_mplr={self.bottleneck_channels_mplr}, dropout={self.dropout_rate}): {self.n_params} params'


class DenseBlock(Module):  # with bottleneck
    """
    Paper: Densely Connected Convolutional Networks
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    """

    def __init__(self, in_channels, growth_rate, n_convs, bottleneck_channels_mplr=4, dropout=0., bias=False):
        self.layers = ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck_channels_mplr, dropout, bias)
            for i in range(n_convs)
        ])
        self.in_channels = in_channels
        self.out_channels = growth_rate * n_convs

    def forward(self, x):
        for layer in self.layers:
            y = layer.forward(x)
            x = torch.cat((y, x), dim=1)  # concatenate the skip connection  (N, C, H, W) -> (N, C + n*k, H, W)
        return x

    def __repr__(self):
        return f'DenseBlock(in_channels={self.in_channels}, out_channels={self.out_channels}): {self.n_params} params'


class DenseTransition(Module):
    """
    Paper: Densely Connected Convolutional Networks
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    """

    def __init__(self, in_channels, out_channels, downsample_by=2, bias=False):
        self.downsample = Sequential(
            BatchNorm2d(in_channels), ReLU(),  # notice the batch norm is applied before the activation
            Conv2d(in_channels,  out_channels, kernel_size=1, padding='same', bias=bias),   # 1x1 conv (reduce channels)
            AvgPool2d(kernel_size=2, stride=downsample_by),                                 # pool /2
        )
        self.compression_factor = out_channels / in_channels
        self.downsampling = downsample_by
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.downsample.forward(x)
        return x

    def __repr__(self):
        return f'DenseTransition(in_channels={self.in_channels}, out_channels={self.out_channels}, downsampling={self.downsampling}): {self.n_params} params'
