import torch
from lib.layers import Module, Linear, Conv2d, AvgPool2d
from lib.functions.activations import softmax, tanh

class LeNet5(Module):  # https://hal.science/hal-03926082/document

    def __init__(self):                                                           # in:  1 x 32 x  32
        self.c1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5)            # ->   6 x 28 x  28
        self.s2 = AvgPool2d(kernel_size=2, stride=2)                              # ->   6 x 14 x  14
        self.c3 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)           # ->  16 x 10 x  10
        self.s4 = AvgPool2d(kernel_size=2, stride=2)                              # ->  16 x  5 x   5
        self.c5 = Conv2d(in_channels=16, out_channels=120, kernel_size=5)         # ->   1 x  1 x 120 (flat)
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
