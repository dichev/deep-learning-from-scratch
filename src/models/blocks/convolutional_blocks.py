import torch
from lib.layers import Module, Sequential, Conv2d, Conv2dGroups, MaxPool2d, BatchNorm2d, ReLU
from lib.functions.activations import relu


class Inception(Module):
    """
    Paper: Going deeper with convolutions
    https://arxiv.org/pdf/1409.4842.pdf?
    """

    def __init__(self, in_channels, out_channels, spec=(0, (0, 0), (0, 0), 0), device='cpu'):
        c1, (c2_reduce, c2), (c3_reduce, c3), c4 = spec
        assert out_channels == c1 + c2 + c3 + c4, f'Wrong channel spec: expected {out_channels} total output channels, but got {c1}+{c2}+{c3}+{c4}={c1 + c2 + c3 + c4}'

        self.branch1 = Sequential(
            Conv2d(in_channels, c1, kernel_size=1, device=device))
        self.branch2 = Sequential(
            Conv2d(in_channels, c2_reduce, kernel_size=1, device=device), ReLU(),
            Conv2d(c2_reduce, c2, kernel_size=3, padding='same', device=device))
        self.branch3 = Sequential(
            Conv2d(in_channels, c3_reduce, kernel_size=1, device=device), ReLU(),
            Conv2d(c3_reduce, c3, kernel_size=5, padding='same', device=device))
        self.branch4 = Sequential(
            MaxPool2d(kernel_size=3, padding='same'),
            Conv2d(in_channels, c4, kernel_size=1, device=device))

    def forward(self, x):
        features = [
            self.branch1.forward(x),
            self.branch2.forward(x),
            self.branch3.forward(x),
            self.branch4.forward(x),
        ]
        x = torch.cat(features, dim=1)  # 4 x (N, C, W, H) ->  (N, 4*C, W, H)
        x = torch.relu(x)  # it might be attached outside the module
        return x



class ResBlock(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, in_channels, out_channels, stride=1, device='cpu'):
        self.downsampled = stride != 1 or in_channels != out_channels

        self.convs = Sequential(
            Conv2d(in_channels,  out_channels, kernel_size=3, padding='same', device=device, stride=stride),
            BatchNorm2d(out_channels, device=device), ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, padding='same', device=device),
            BatchNorm2d(out_channels, device=device),  # no ReLU
        )
        if self.downsampled:
            self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        y = self.convs.forward(x)
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

    def __init__(self, in_channels, mid_channels, out_channels, stride=1, device='cpu'):
        assert mid_channels == out_channels // 4  # just following the models in the paper
        self.downsampled = stride != 1 or in_channels != out_channels

        self.convs = Sequential(
            Conv2d(in_channels,  mid_channels, kernel_size=1, padding='same', device=device, stride=stride),   # 1x1 conv (reduce channels)
            BatchNorm2d(mid_channels, device=device), ReLU(),
            Conv2d(mid_channels, mid_channels, kernel_size=3, padding='same', device=device),                  # 3x3 conv
            BatchNorm2d(mid_channels, device=device), ReLU(),
            Conv2d(mid_channels, out_channels, kernel_size=1, padding='same', device=device),                  # 1x1 conv (expand channels)
            BatchNorm2d(out_channels, device=device),  # no ReLU
        )
        if self.downsampled:
            self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, device=device)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        y = self.convs.forward(x)
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

    def __init__(self, in_channels, mid_channels, out_channels, groups, stride=1, device='cpu'):
        assert mid_channels == out_channels // 2  # just following the models in the paper
        self.downsampled = stride != 1 or in_channels != out_channels

        self.convs = Sequential(
            Conv2d(in_channels,  mid_channels, kernel_size=1, padding='same', device=device, stride=stride),        # 1x1 conv (reduce channels)
            BatchNorm2d(mid_channels, device=device), ReLU(),
            Conv2dGroups(mid_channels, mid_channels, kernel_size=3, padding='same', device=device, groups=groups),  # groups x 3x3 conv with channels/groups
            BatchNorm2d(mid_channels, device=device), ReLU(),
            Conv2d(mid_channels, out_channels, kernel_size=1, padding='same', device=device),                       # 1x1 conv (expand channels)
            BatchNorm2d(out_channels, device=device),  # no ReLU
        )
        if self.downsampled:
            self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, device=device)

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride

    def forward(self, x):
        y = self.convs.forward(x)
        if self.downsampled:
            x = self.project.forward(x)
        y = relu(y + x)  # add the skip connection
        return y

    def __repr__(self):
        return f'ResNeXtBlock(in_channels={self.in_channels}, mid_channels={self.in_channels}, out_channels={self.out_channels}, groups={self.groups}, stride={self.stride}): {self.n_params} params'

