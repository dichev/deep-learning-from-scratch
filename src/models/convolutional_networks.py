import torch
import torch.nn.functional as F
from lib.layers import Module, Sequential, Linear, Conv2d, AvgPool2d, MaxPool2d, Dropout, LocalResponseNorm
from lib.functions.activations import softmax, relu, tanh
from utils.other import conv2d_calc_out_size


class SimpleCNN(Module):
    def __init__(self, device='cpu'):                                                          # in:  3, 32,  32
        self.conv1 = Conv2d(in_channels=3, out_channels=6,  kernel_size=5, device=device)       # ->   6, 28,  28
        self.pool1 = MaxPool2d(kernel_size=2, stride=2, device=device)                          # ->   6, 14,  14
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, device=device)       # ->  16, 10,  10
        self.pool2 = MaxPool2d(kernel_size=2, stride=2, device=device)                          # ->   6,  5,   5
        self.fc1 = Linear(input_size=16 * 5 * 5, output_size=120, device=device)                # ->   1,  1, 120 (flat)
        self.fc2 = Linear(input_size=120, output_size=84, device=device)                        # ->  84
        self.fc3 = Linear(input_size=84, output_size=10, device=device)                         # ->  10

        self.device = device

    def forward(self, x):
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = relu(x)
        x = self.pool2.forward(x)
        x = x.flatten(start_dim=1)
        x = self.fc1.forward(x)
        x = relu(x)
        x = self.fc2.forward(x)
        x = relu(x)
        x = self.fc3.forward(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=10):
        x = torch.randn(n_samples, 3, 32, 32).to(self.device)
        return self.forward(x)

class LeNet5(Module):
    """
    Paper: Gradient-based learning applied to document recognition
    https://hal.science/hal-03926082/document
    """

    def __init__(self, device='cpu'):                                                      # in:  1, 32,  32
        self.c1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5, device=device)      # ->   6, 28,  28
        self.s2 = AvgPool2d(kernel_size=2, stride=2, device=device)                        # ->   6, 14,  14
        self.c3 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, device=device)     # ->  16, 10,  10
        self.s4 = AvgPool2d(kernel_size=2, stride=2, device=device)                        # ->  16,  5,   5
        self.c5 = Conv2d(in_channels=16, out_channels=120, kernel_size=5, device=device)   # ->   1,  1, 120 (flat)
        self.f6 = Linear(input_size=120, output_size=84, device=device)                    # ->  84
        self.f7 = Linear(input_size=84, output_size=10, device=device)                     # ->  10

        self.device = device

    def forward(self, x):
        N, C, W, H = x.shape
        assert (C, W, H) == (1, 32, 32), f'Expected input shape {(1, 32, 32)} but got {(C, W, H)}'

        A, S = 1.7159, 2/3

        x = self.c1.forward(x)
        x = A * tanh(S * x)
        x = self.s2.forward(x)
        x = self.c3.forward(x)
        x = A * tanh(S * x)
        x = self.s4.forward(x)
        x = self.c5.forward(x)
        x = A * tanh(S * x)

        x = x.flatten(start_dim=1)
        x = self.f6.forward(x)
        x = A * tanh(S * x)
        x = self.f7.forward(x)
        # p = softmax(x)  # @ the paper work used Euclidean RBF units with parameters fixed by hand

        return x

    @torch.no_grad()
    def test(self, n_samples=10):
        x = torch.randn(n_samples, 1, 32, 32).to(self.device)
        return self.forward(x)



class AlexNet(Module):
    """
    Paper: ImageNet Classification with Deep ConvolutionalNeural Networks
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    * Following the paper, but modified for a single GPU
    """

    def __init__(self, n_classes=1000, device='cpu'):

        self.features = Sequential(                                                                                      # in:  3, 227, 227
            Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0, device=device), relu,            # ->  96, 55, 55
            LocalResponseNorm(size=5, alpha=5*1e-4, beta=.75, k=2.),
            MaxPool2d(kernel_size=3, stride=2, device=device),                                                           # ->  96, 27, 27 (max)
            Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2, device=device), relu,                     # -> 256, 27, 27
            LocalResponseNorm(size=5, alpha=5*1e-4, beta=.75, k=2.),
            MaxPool2d(kernel_size=3, stride=2, device=device),                                                           # -> 256, 13, 13 (max)
            Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1, device=device), relu,                    # -> 384, 13, 13
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, device=device), relu,                    # -> 256, 13, 13
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, device=device), relu,                    # -> 256, 13, 13
            MaxPool2d(kernel_size=3, stride=2, device=device),                                                           # -> 256,  6,  6 (max)
        )
        self.classifier = Sequential(                                                                                    # -> 9216 (flatten)
           Dropout(0.5),
           Linear(input_size=256*6*6, output_size=4096, device=device),  relu,                                           # -> 4096
           Dropout(0.5),
           Linear(input_size=4096, output_size=4096, device=device), relu,                                               # -> 4096
           Linear(input_size=4096, output_size=n_classes, device=device),                                                # -> n_classes (e.g. 1000)
        )
        self.device = device

    def forward(self, x, verbose=False):
        N, C, W, H = x.shape
        assert (C, W, H) == (3, 227, 227), f'Expected input shape {(3, 227, 227)} but got {(C, W, H)}'

        x = self.features.forward(x, verbose)
        x = x.flatten(start_dim=1)
        x = self.classifier.forward(x, verbose)
        x = softmax(x)  # @ in the paper were actually used "1000 independent logistic units" to avoid calculating the normalization factor
        return x

    @torch.no_grad()
    def test(self, n_samples=10):
        x = torch.randn(n_samples, 3, 227, 227).to(self.device)
        return self.forward(x, verbose=True)


class VGG16(Module):
    """
    Paper: Very Deep Convolutional Networks fpr Large-Scale Image Recognition
    https://arxiv.org/pdf/1409.1556.pdf
    """

    def __init__(self, n_classes=1000, device='cpu'):
        self.features = Sequential(                                                                                          # in:  3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding='same', device=device), relu,            # ->  64, 224, 224
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same', device=device), relu,           # ->  64, 224, 224
            MaxPool2d(kernel_size=2, stride=2, device=device),                                                               # ->  64, 112, 112 (max)
            Conv2d(in_channels=64,  out_channels=128, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 128, 112, 112
            Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 128, 112, 112
            MaxPool2d(kernel_size=2, stride=2, device=device),                                                               # -> 128,  56,  56 (max)
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 256,  56,  56
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 256,  56,  56
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 256,  56,  56
            MaxPool2d(kernel_size=2, stride=2, device=device),                                                               # -> 256,  28,  28 (max)
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 512,  28,  28
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 512,  28,  28
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 512,  28,  28
            MaxPool2d(kernel_size=2, stride=2, device=device),                                                               # -> 512,  14,  14 (max)
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 512,  14,  14
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 512,  14,  14
            Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding='same', device=device), relu,         # -> 512,  14,  14
            MaxPool2d(kernel_size=2, stride=2, device=device),                                                               # -> 512,   7,   7 (max)
        )
        self.classifier = Sequential(                                                                                        # -> 9216 (flatten)
           Linear(input_size=512*7*7, output_size=4096, device=device),  relu,                                           # -> 4096
           Dropout(0.5),
           Linear(input_size=4096, output_size=4096, device=device), relu,                                               # -> 4096
           Dropout(0.5),
           Linear(input_size=4096, output_size=n_classes, device=device),                                                # -> n_classes (e.g. 1000)
        )
        self.device = device

    def forward(self, x, verbose=False):
        N, C, W, H = x.shape
        assert (C, W, H) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, W, H)}'

        x = self.features.forward(x, verbose)
        x = x.flatten(start_dim=1)
        x = self.classifier.forward(x, verbose)
        x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=10):
        x = torch.randn(n_samples, 3, 224, 224).to(self.device)
        return self.forward(x, verbose=True)

