import torch
from lib.layers import Module, Sequential, Linear, Conv2d, Conv2dGroups, AvgPool2d, MaxPool2d, BatchNorm2d, ReLU, Flatten
from models.blocks.convolutional_blocks import ResBlock, ResBottleneckBlock, ResNeXtBlock

class ResNet34(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_classes=1000, attention=False, device='cpu'):

        self.stem = Sequential(                                                                                   # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', device=device),       # ->   64, 112, 112
            BatchNorm2d(64, device=device), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1), device=device),                              # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ResBlock(in_channels=64, out_channels=64, attention=attention, device=device),                    # ->   64,  56,  56 (stride /2)
            ResBlock(in_channels=64, out_channels=64, attention=attention, device=device),                    # ->   64,  56,  56
            ResBlock(in_channels=64, out_channels=64, attention=attention, device=device),                    # ->   64,  56,  56

            ResBlock(in_channels=64, out_channels=128, attention=attention, device=device, stride=2),        # ->  128,  28,  28 (stride /2)
            ResBlock(in_channels=128, out_channels=128, attention=attention, device=device),                  # ->  128,  28,  28
            ResBlock(in_channels=128, out_channels=128, attention=attention, device=device),                  # ->  128,  28,  28
            ResBlock(in_channels=128, out_channels=128, attention=attention, device=device),                  # ->  128,  28,  28

            ResBlock(in_channels=128, out_channels=256, attention=attention, device=device, stride=2),        # ->  256,  14,  14 (stride /2)
            ResBlock(in_channels=256, out_channels=256, attention=attention, device=device),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention, device=device),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention, device=device),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention, device=device),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention, device=device),                  # ->  256,  14,  14

            ResBlock(in_channels=256, out_channels=512, attention=attention, device=device, stride=2),        # ->  512,   7,   7 (stride /2)
            ResBlock(in_channels=512, out_channels=512, attention=attention, device=device),                  # ->  512,   7,   7
            ResBlock(in_channels=512, out_channels=512, attention=attention, device=device),                  # ->  512,   7,   7
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

    def __init__(self, n_classes=1000, attention=False, device='cpu'):

        self.stem = Sequential(                                                                                   # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', device=device),       # ->   64, 112, 112
            BatchNorm2d(64, device=device), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1), device=device),                              # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ResBottleneckBlock(in_channels=64,  mid_channels=64, out_channels=256, attention=attention, device=device),                 # 64 ->  [64] -> 256,  56,  56 (stride /2)
            ResBottleneckBlock(in_channels=256, mid_channels=64, out_channels=256, attention=attention, device=device),                # 256 ->  [64] -> 256,  56,  56
            ResBottleneckBlock(in_channels=256, mid_channels=64, out_channels=256, attention=attention, device=device),                # 256 ->  [64] -> 256,  56,  56

            ResBottleneckBlock(in_channels=256, mid_channels=128, out_channels=512, attention=attention, device=device, stride=2),     # 256 -> [128] -> 512,  28,  28 (stride /2)
            ResBottleneckBlock(in_channels=512, mid_channels=128, out_channels=512, attention=attention, device=device),               # 512 -> [128] -> 512,  28,  28
            ResBottleneckBlock(in_channels=512, mid_channels=128, out_channels=512, attention=attention, device=device),               # 512 -> [128] -> 512,  28,  28
            ResBottleneckBlock(in_channels=512, mid_channels=128, out_channels=512, attention=attention, device=device),               # 512 -> [128] -> 512,  28,  28

            ResBottleneckBlock(in_channels=512,  mid_channels=256, out_channels=1024, attention=attention, device=device, stride=2),   # 512 -> [256] -> 1024, 28,  28 (stride /2)
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention, device=device),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention, device=device),            # 1024 -> [256] -> 1024, 28,  28

            ResBottleneckBlock(in_channels=1024, mid_channels=512, out_channels=2048, attention=attention, device=device, stride=2),  # 1024 -> [512] -> 2048,  7,   7 (stride /2)
            ResBottleneckBlock(in_channels=2048, mid_channels=512, out_channels=2048, attention=attention, device=device),            # 2048 -> [512] -> 2048,  7,   7
            ResBottleneckBlock(in_channels=2048, mid_channels=512, out_channels=2048, attention=attention, device=device),            # 2048 -> [512] -> 2048,  7,   7
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


class ResNeXt50(Module):  # same computations/params as ResNet-50, but more channels and better accuracy
    """
    Paper: Aggregated Residual Transformations for Deep Neural Networks
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf
    """

    def __init__(self, n_classes=1000, attention=False, device='cpu'):

        self.stem = Sequential(                                                                                   # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', device=device),       # ->   64, 112, 112
            BatchNorm2d(64, device=device), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1), device=device),                              # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ResNeXtBlock(in_channels=64,  mid_channels=128, out_channels=256, groups=32, attention=attention, device=device),                 # 64 ->  [128] -> 256,  56,  56 (stride /2)
            ResNeXtBlock(in_channels=256, mid_channels=128, out_channels=256, groups=32, attention=attention, device=device),                # 256 ->  [128] -> 256,  56,  56
            ResNeXtBlock(in_channels=256, mid_channels=128, out_channels=256, groups=32, attention=attention, device=device),                # 256 ->  [128] -> 256,  56,  56

            ResNeXtBlock(in_channels=256, mid_channels=256, out_channels=512, groups=32, attention=attention, device=device, stride=2),      # 256 ->  [256] -> 512,  28,  28 (stride /2)
            ResNeXtBlock(in_channels=512, mid_channels=256, out_channels=512, groups=32, attention=attention, device=device),                # 512 ->  [256] -> 512,  28,  28
            ResNeXtBlock(in_channels=512, mid_channels=256, out_channels=512, groups=32, attention=attention, device=device),                # 512 ->  [256] -> 512,  28,  28
            ResNeXtBlock(in_channels=512, mid_channels=256, out_channels=512, groups=32, attention=attention, device=device),                # 512 ->  [256] -> 512,  28,  28

            ResNeXtBlock(in_channels=512,  mid_channels=512, out_channels=1024, groups=32, attention=attention, device=device, stride=2),    # 512 ->  [512] -> 1024, 28,  28 (stride /2)
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention, device=device),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention, device=device),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention, device=device),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention, device=device),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention, device=device),             # 1024 ->  [512] -> 1024, 28,  28

            ResNeXtBlock(in_channels=1024, mid_channels=1024, out_channels=2048, groups=32, attention=attention, device=device, stride=2),  # 1024 -> [1024] -> 2048,  7,   7 (stride /2)
            ResNeXtBlock(in_channels=2048, mid_channels=1024, out_channels=2048, groups=32, attention=attention, device=device),            # 2048 -> [1024] -> 2048,  7,   7
            ResNeXtBlock(in_channels=2048, mid_channels=1024, out_channels=2048, groups=32, attention=attention, device=device),            # 2048 -> [1024] -> 2048,  7,   7
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


class SEResNet50(ResNet50):
    """
    Paper: Squeeze-and-Excitation Layer
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, n_classes=1000, device='cpu'):
        super().__init__(n_classes, attention=True, device=device)


class SEResNeXt50(ResNeXt50):
    """
    Paper: Squeeze-and-Excitation Layer
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, n_classes=1000, device='cpu'):
        super().__init__(n_classes, attention=True, device=device)

