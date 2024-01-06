import torch
from lib.layers import Module, Sequential, Linear, Conv2d, AvgPool2d, MaxPool2d, BatchNorm2d, ReLU, Flatten
from lib.functions.activations import relu


class Residual(Module):
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
        return f'Residual(in_channels={self.in_channels}, out_channels={self.out_channels}, stride={self.stride}): {self.n_params} params'


class ResidualBottleneck(Module):
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
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        y = self.convs.forward(x)
        if self.downsampled:
            x = self.project.forward(x)
        y = relu(y + x)  # add the skip connection
        return y

    def __repr__(self):
        return f'ResidualBottleneck(in_channels={self.in_channels}, out_channels={self.out_channels}, stride={self.stride}): {self.n_params} params'


class ResNet34(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_classes=1000, device='cpu'):

        self.stem = Sequential(                                                                                   # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', device=device),       # ->   64, 112, 112
            BatchNorm2d(64, device=device), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1), device=device),                              # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            Residual(in_channels=64, out_channels=64, device=device),                    # ->   64,  56,  56 (stride /2)
            Residual(in_channels=64, out_channels=64, device=device),                    # ->   64,  56,  56
            Residual(in_channels=64, out_channels=64, device=device),                    # ->   64,  56,  56

            Residual(in_channels=64,  out_channels=128, device=device, stride=2),        # ->  128,  28,  28 (stride /2)
            Residual(in_channels=128, out_channels=128, device=device),                  # ->  128,  28,  28
            Residual(in_channels=128, out_channels=128, device=device),                  # ->  128,  28,  28
            Residual(in_channels=128, out_channels=128, device=device),                  # ->  128,  28,  28

            Residual(in_channels=128, out_channels=256, device=device, stride=2),        # ->  256,  14,  14 (stride /2)
            Residual(in_channels=256, out_channels=256, device=device),                  # ->  256,  14,  14
            Residual(in_channels=256, out_channels=256, device=device),                  # ->  256,  14,  14
            Residual(in_channels=256, out_channels=256, device=device),                  # ->  256,  14,  14
            Residual(in_channels=256, out_channels=256, device=device),                  # ->  256,  14,  14
            Residual(in_channels=256, out_channels=256, device=device),                  # ->  256,  14,  14

            Residual(in_channels=256, out_channels=512, device=device, stride=2),        # ->  512,   7,   7 (stride /2)
            Residual(in_channels=512, out_channels=512, device=device),                  # ->  512,   7,   7
            Residual(in_channels=512, out_channels=512, device=device),                  # ->  512,   7,   7
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                     # -> 512, 1, 1
           Flatten(),                                                                    # -> 512
           Linear(input_size=512, output_size=n_classes, device=device)                  # -> n_classes(1000)
        )
        self.device = device

    def forward(self, x, verbose=False):
        N, C, W, H = x.shape
        assert (C, W, H) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, W, H)}'

        x = self.stem.forward(x, verbose)
        x = self.body.forward(x, verbose)
        x = self.head.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 224, 224).to(self.device)
        return self.forward(x, verbose=True)


class ResNet50(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_classes=1000, device='cpu'):

        self.stem = Sequential(                                                                                   # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', device=device),       # ->   64, 112, 112
            BatchNorm2d(64, device=device), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1), device=device),                              # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ResidualBottleneck(in_channels=64,  mid_channels=64, out_channels=256, device=device),                 # 64 ->  [64] -> 256,  56,  56 (stride /2)
            ResidualBottleneck(in_channels=256, mid_channels=64, out_channels=256, device=device),                # 256 ->  [64] -> 256,  56,  56
            ResidualBottleneck(in_channels=256, mid_channels=64, out_channels=256, device=device),                # 256 ->  [64] -> 256,  56,  56

            ResidualBottleneck(in_channels=256, mid_channels=128, out_channels=512, device=device, stride=2),     # 256 -> [128] -> 512,  28,  28 (stride /2)
            ResidualBottleneck(in_channels=512, mid_channels=128, out_channels=512, device=device),               # 512 -> [128] -> 512,  28,  28
            ResidualBottleneck(in_channels=512, mid_channels=128, out_channels=512, device=device),               # 512 -> [128] -> 512,  28,  28
            ResidualBottleneck(in_channels=512, mid_channels=128, out_channels=512, device=device),               # 512 -> [128] -> 512,  28,  28

            ResidualBottleneck(in_channels=512,  mid_channels=256, out_channels=1024, device=device, stride=2),   # 512 -> [256] -> 1024, 28,  28 (stride /2)
            ResidualBottleneck(in_channels=1024, mid_channels=256, out_channels=1024, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResidualBottleneck(in_channels=1024, mid_channels=256, out_channels=1024, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResidualBottleneck(in_channels=1024, mid_channels=256, out_channels=1024, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResidualBottleneck(in_channels=1024, mid_channels=256, out_channels=1024, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResidualBottleneck(in_channels=1024, mid_channels=256, out_channels=1024, device=device),            # 1024 -> [256] -> 1024, 28,  28

            ResidualBottleneck(in_channels=1024, mid_channels=512, out_channels=2048, device=device, stride=2),  # 1024 -> [512] -> 2048,  7,   7 (stride /2)
            ResidualBottleneck(in_channels=2048, mid_channels=512, out_channels=2048, device=device),            # 2048 -> [512] -> 2048,  7,   7
            ResidualBottleneck(in_channels=2048, mid_channels=512, out_channels=2048, device=device),            # 2048 -> [512] -> 2048,  7,   7
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                                             # -> 2048, 1, 1
           Flatten(),                                                                                            # -> 2048
           Linear(input_size=2048, output_size=n_classes, device=device)                                         # -> n_classes(1000)
        )
        self.device = device

    def forward(self, x, verbose=False):
        N, C, W, H = x.shape
        assert (C, W, H) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, W, H)}'

        x = self.stem.forward(x, verbose)
        x = self.body.forward(x, verbose)
        x = self.head.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 224, 224).to(self.device)
        return self.forward(x, verbose=True)
