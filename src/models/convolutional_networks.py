import torch
from lib.layers import Module, Sequential, Linear, Conv2d, AvgPool2d, MaxPool2d, Dropout
from lib.functions.activations import softmax, relu, tanh
from utils.other import conv2d_calc_out_size


class LeNet5(Module):
    """
    Paper: "Gradient-based learning applied to document recognition"
    https://hal.science/hal-03926082/document
    """

    def __init__(self):                                                           # in:  1, 32,  32
        self.c1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5)            # ->   6, 28,  28
        self.s2 = AvgPool2d(kernel_size=2, stride=2)                              # ->   6, 14,  14
        self.c3 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)           # ->  16, 10,  10
        self.s4 = AvgPool2d(kernel_size=2, stride=2)                              # ->  16,  5,   5
        self.c5 = Conv2d(in_channels=16, out_channels=120, kernel_size=5)         # ->   1,  1, 120 (flat)
        self.f6 = Linear(input_size=120, output_size=84)                          # ->  84
        self.f7 = Linear(input_size=84, output_size=10)                           # ->  10

    def forward(self, X):
        A, S = 1.7159, 2/3

        X = self.c1.forward(X)
        X = A*tanh(S*X)
        X = self.s2.forward(X)
        X = self.c3.forward(X)
        X = A*tanh(S*X)
        X = self.s4.forward(X)
        X = self.c5.forward(X)
        X = A*tanh(S*X)

        X = X.flatten(start_dim=1)
        X = self.f6.forward(X)
        X = A*tanh(S*X)
        X = self.f7.forward(X)
        p = softmax(X)  # @ the paper work used Euclidean RBF units with parameters fixed by hand

        return p


class AlexNet(Module):
    """
    Paper: "ImageNet Classification with Deep ConvolutionalNeural Networks"
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    * Following the paper, but for a single GPU
    """

    def __init__(self, n_classes=1000):

        self.features = Sequential(                                                               # in:  3, 227, 227
            Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0), relu,    # ->  96, 55, 55
            MaxPool2d(kernel_size=3, stride=2),                                                   # ->  96, 27, 27 (max)
            Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), relu,             # -> 256, 27, 27
            MaxPool2d(kernel_size=3, stride=2),                                                   # -> 256, 13, 13 (max)
            Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), relu,            # -> 384, 13, 13
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), relu,            # -> 256, 13, 13
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), relu,            # -> 256, 13, 13
            MaxPool2d(kernel_size=3, stride=2),                                                   # -> 256,  6,  6 (max)
        )

        self.classifier = Sequential(                                                             # -> 9216 (flatten)
           Dropout(0.5),
           Linear(input_size=256*6*6, output_size=4096),  relu,                                   # -> 4096
           Dropout(0.5),
           Linear(input_size=4096, output_size=4096), relu,                                       # -> 4096
           Linear(input_size=4096, output_size=n_classes),                                        # -> n_classes (e.g. 1000)
        )

        # todo: local response normalization

    def forward(self, x, verbose=False):
        x = self.features.forward(x, verbose)
        x = x.flatten(start_dim=1)
        x = self.classifier.forward(x, verbose)
        x = softmax(x)  # @ in the paper were actually used "1000 independent logistic units to avoid calculating the normalization factor
        return x
