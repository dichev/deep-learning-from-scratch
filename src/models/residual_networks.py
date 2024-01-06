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
        if self.downsampled:
            self.project = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, device=device)

        self.c1 = Conv2d(in_channels, out_channels, kernel_size=3, padding='same', stride=stride, device=device)
        self.bn1 = BatchNorm2d(out_channels, device=device)
        self.c2 = Conv2d(out_channels, out_channels, kernel_size=3, padding='same', device=device)
        self.bn2 = BatchNorm2d(out_channels, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        y = self.bn1.forward(self.c1.forward(x))
        y = relu(y)
        if self.downsampled:
            x = self.project.forward(x)
        y = self.bn2.forward(self.c2.forward(y)) + x  # the skip connection
        y = relu(y)
        return y

    def __repr__(self):
        return f'Residual(in_channels={self.in_channels}, out_channels={self.out_channels}, stride={self.stride}): {self.n_params} params'

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
            Residual(in_channels=64, out_channels=64, device=device),                    # ->   64,  56,  56
            Residual(in_channels=64, out_channels=64, device=device),
            Residual(in_channels=64, out_channels=64, device=device),

            Residual(in_channels=64,  out_channels=128, device=device, stride=2),        # ->  128,  28,  28
            Residual(in_channels=128, out_channels=128, device=device),
            Residual(in_channels=128, out_channels=128, device=device),
            Residual(in_channels=128, out_channels=128, device=device),

            Residual(in_channels=128, out_channels=256, device=device, stride=2),        # ->  256,  14,  14
            Residual(in_channels=256, out_channels=256, device=device),
            Residual(in_channels=256, out_channels=256, device=device),
            Residual(in_channels=256, out_channels=256, device=device),
            Residual(in_channels=256, out_channels=256, device=device),
            Residual(in_channels=256, out_channels=256, device=device),

            Residual(in_channels=256, out_channels=512, device=device, stride=2),         # ->  512,   7,   7
            Residual(in_channels=512, out_channels=512, device=device),
            Residual(in_channels=512, out_channels=512, device=device),
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                      # -> 512, 1, 1
           Flatten(),                                                                     # -> 512
           Linear(input_size=512, output_size=n_classes, device=device)                   # -> n_classes(1000)
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
