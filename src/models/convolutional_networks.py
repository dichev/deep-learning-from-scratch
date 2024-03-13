import torch
from lib.layers import Module, Sequential, Linear, Conv2d, AvgPool2d, MaxPool2d, BatchNorm2d, Dropout, LocalResponseNorm, ReLU, Flatten
from lib.functions.activations import relu, tanh
from models.blocks.convolutional_blocks import Inception


class SimpleCNN(Module):
    def __init__(self, n_classes=10):                                            # in:  3, 32,  32
        self.conv1 = Conv2d(in_channels=3, out_channels=6,  kernel_size=5)       # ->   6, 28,  28
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)                          # ->   6, 14,  14
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)       # ->  16, 10,  10
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)                          # ->  16,  5,   5
        self.fc1 = Linear(input_size=16 * 5 * 5, output_size=120)                # ->  120 (flat)
        self.fc2 = Linear(input_size=120, output_size=84)                        # ->  84
        self.fc3 = Linear(input_size=84, output_size=n_classes)                  # ->  n_classes(10)

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
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 32, 32, device=self.device_of_first_parameter())
        return self.forward(x)


class SimpleFullyCNN(Module):  # can be used to "convolve" the classifier across a larger image, and then average out the class
    def __init__(self, n_classes=10):                                            # in:  3, 32,  32
        self.conv1 = Conv2d(in_channels=3, out_channels=6,  kernel_size=5)       # ->   6, 28,  28
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)                          # ->   6, 14,  14
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)       # ->  16, 10,  10
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)                          # ->  16,  5,   5
        # classifier: convert fully-connected layers to convolutional:
        self.conv3 = Conv2d(in_channels=16, out_channels=120, kernel_size=5)     # ->   1,  1, 120
        self.conv4 = Conv2d(in_channels=120, out_channels=84, kernel_size=1)     # ->   1,  1, 84
        self.conv5 = Conv2d(in_channels=84, out_channels=10, kernel_size=1)      # ->   1,  1, n_classes(10)

        self.n_classes = n_classes

    def forward(self, x):
        N, C, W, H = x.shape
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = relu(x)
        x = self.pool2.forward(x)
        x = self.conv3.forward(x)
        x = relu(x)
        x = self.conv4.forward(x)
        x = relu(x)
        x = self.conv5.forward(x)
        x = x.view(N, self.n_classes)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 32, 32, device=self.device_of_first_parameter())
        return self.forward(x)

class LeNet5(Module):
    """
    Paper: Gradient-based learning applied to document recognition
    https://hal.science/hal-03926082/document
    """

    def __init__(self, n_classes=10):                                       # in:  1, 32,  32
        self.c1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5)      # ->   6, 28,  28
        self.s2 = AvgPool2d(kernel_size=2, stride=2)                        # ->   6, 14,  14
        self.c3 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)     # ->  16, 10,  10
        self.s4 = AvgPool2d(kernel_size=2, stride=2)                        # ->  16,  5,   5
        self.c5 = Conv2d(in_channels=16, out_channels=120, kernel_size=5)   # ->   1,  1, 120 (flat)
        self.f6 = Linear(input_size=120, output_size=84)                    # ->  84
        self.f7 = Linear(input_size=84, output_size=n_classes)              # ->  # ->  n_classes(10)

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
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 1, 32, 32, device=self.device_of_first_parameter())
        return self.forward(x)


class AlexNet(Module):
    """
    Paper: ImageNet Classification with Deep Convolutional Neural Networks
    https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    * Following the paper, but modified for a single GPU
    """

    def __init__(self, n_classes=1000):

        self.features = Sequential(                                                                       # in:  3, 227, 227
            Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0), ReLU(),          # ->  96, 55, 55
            LocalResponseNorm(size=5, alpha=5*1e-4, beta=.75, k=2.),
            MaxPool2d(kernel_size=3, stride=2),                                                           # ->  96, 27, 27 (max)
            Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), ReLU(),                   # -> 256, 27, 27
            LocalResponseNorm(size=5, alpha=5*1e-4, beta=.75, k=2.),
            MaxPool2d(kernel_size=3, stride=2),                                                           # -> 256, 13, 13 (max)
            Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), ReLU(),                  # -> 384, 13, 13
            Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), ReLU(),                  # -> 256, 13, 13
            Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), ReLU(),                  # -> 256, 13, 13
            MaxPool2d(kernel_size=3, stride=2),                                                           # -> 256,  6,  6 (max)
        )
        self.classifier = Sequential(
            Flatten(),                                                                                    # -> 9216 (flatten)
            Dropout(0.5),
            Linear(input_size=256*6*6, output_size=4096), ReLU(),                                         # -> 4096
            Dropout(0.5),
            Linear(input_size=4096, output_size=4096), ReLU(),                                            # -> 4096
            Linear(input_size=4096, output_size=n_classes),                                               # -> n_classes(1000)
        )

    def forward(self, x, verbose=False):
        N, C, W, H = x.shape
        assert (C, W, H) == (3, 227, 227), f'Expected input shape {(3, 227, 227)} but got {(C, W, H)}'

        x = self.features.forward(x, verbose)
        x = self.classifier.forward(x, verbose)
        # x = softmax(x)  # @ in the paper were actually used "1000 independent logistic units" to avoid calculating the normalization factor
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 227, 227, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


class NetworkInNetwork(Module):
    """
    Paper: Network In Network
    https://arxiv.org/pdf/1312.4400.pdf
    """

    def __init__(self, n_classes=1000):

        def MLPConv(in_channels, out_channels, kernel_size, stride=1, padding=0):
            return Sequential(
                Conv2d(in_channels, out_channels, kernel_size, stride, padding), ReLU(),
                Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding='same'), ReLU(),  # 1x1 convolution == fully connected layer, which acts independently on each pixel location
                Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding='same'), ReLU(),  # 1x1 convolution
            )

        # the convolution parameters are based on AlexNet
        self.classifier = Sequential(                                                     # in:  3, 227, 227
            MLPConv(in_channels=3,   out_channels=96 , kernel_size=11,  stride=4),        # ->  96, 55, 55
            Dropout(0.5),
            MaxPool2d(kernel_size=3, stride=2),                                           # ->  96, 27, 27 (max)

            MLPConv(in_channels=96,  out_channels=256, kernel_size=5,  padding=2),        # -> 256, 27, 27
            Dropout(0.5),
            MaxPool2d(kernel_size=3, stride=2),                                           # -> 256, 13, 13 (max)

            MLPConv(in_channels=256, out_channels=384, kernel_size=3,  padding=1),        # -> 384, 13, 13
            Dropout(0.5),
            MaxPool2d(kernel_size=3, stride=2),                                           # -> 384,  6,  6 (max)

            MLPConv(in_channels=384, out_channels=n_classes, kernel_size=3,  padding=1),  # -> n_classes(1000), 6, 6  # @ in the paper it seems they used just 3 MPLConv blocks
            AvgPool2d(kernel_size=6),                                                     # -> n_classes(1000), 1, 1
            Flatten()                                                                     # -> n_classes(1000)
        )

    def forward(self, x, verbose=False):
        N, C, W, H = x.shape
        assert (C, W, H) == (3, 227, 227), f'Expected input shape {(3, 227, 227)} but got {(C, W, H)}'

        x = self.classifier.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 227, 227, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


class VGG16(Module):
    """
    Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/pdf/1409.1556.pdf
    """

    def __init__(self, n_classes=1000):

        def ConvBlock(n_convs, in_channels, out_channels):
            block = Sequential()
            for i in range(n_convs):
                block.add(Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding='same'))
                block.add(ReLU())
            block.add(MaxPool2d(kernel_size=2, stride=2))
            return block

        self.features = Sequential(                                              # in:   3, 224, 224
            ConvBlock(n_convs=2, in_channels=3,   out_channels=64),              # ->   64, 112, 112
            ConvBlock(n_convs=2, in_channels=64,  out_channels=128),             # ->  128,  56,  56
            ConvBlock(n_convs=3, in_channels=128, out_channels=256),             # ->  256,  28,  28
            ConvBlock(n_convs=3, in_channels=256, out_channels=512),             # ->  512,  14,  14
            ConvBlock(n_convs=3, in_channels=512, out_channels=512),             # ->  512,   7,   7
        )
        self.classifier = Sequential(
           Flatten(),                                                            # -> 9216 (flatten)
           Linear(input_size=512*7*7, output_size=4096), ReLU(),                 # -> 4096
           Dropout(0.5),
           Linear(input_size=4096, output_size=4096), ReLU(),                    # -> 4096
           Dropout(0.5),
           Linear(input_size=4096, output_size=n_classes),                       # -> n_classes(1000)
        )

    def forward(self, x, verbose=False):
        N, C, W, H = x.shape
        assert (C, W, H) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, W, H)}'

        x = self.features.forward(x, verbose)
        x = self.classifier.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 224, 224, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


class GoogLeNet(Module):  # Inception modules
    """
    Paper: Going deeper with convolutions
    https://arxiv.org/pdf/1409.4842.pdf?
    """

    def __init__(self, n_classes=1000):
        self.stem = Sequential(                                                                         # in:  3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same'), ReLU(),    # ->  64, 112, 112
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),  # (left, right, top, bottom)     # ->  64,  56,  56
            LocalResponseNorm(size=5, alpha=5 * 1e-4, beta=.75, k=2.),
            Conv2d(in_channels=64, out_channels=64,  kernel_size=1), ReLU(),                            # ->  64,  56,  56
            Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding='same'), ReLU(),  # -> 192,  56,  56
            LocalResponseNorm(size=5, alpha=5 * 1e-4, beta=.75, k=2.),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                                   # -> 192,  28,  28
        )
        self.body = Sequential(  # @paper: without the auxiliary classifiers in the intermediate layers
            Inception(in_channels=192, out_channels=256,  spec=( 64,  (96, 128), (16,  32),  32)),      # ->  256, 28, 28   159K 128M  inception (3a)
            Inception(in_channels=256, out_channels=480,  spec=(128, (128, 192), (32,  96),  64)),      # ->  480, 28, 28   380K 304M  inception (3b)
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                                   # ->  480, 14, 14  (max)
            Inception(in_channels=480, out_channels=512,  spec=(192,  (96, 208), (16,  48),  64)),      # ->  512, 14, 14   364K 73M   inception (4a)
            Inception(in_channels=512, out_channels=512,  spec=(160, (112, 224), (24,  64),  64)),      # ->  512, 14, 14   437K 88M   inception (4b)
            Inception(in_channels=512, out_channels=512,  spec=(128, (128, 256), (24,  64),  64)),      # ->  512, 14, 14   463K 100M  inception (4c)
            Inception(in_channels=512, out_channels=528,  spec=(112, (144, 288), (32,  64),  64)),      # ->  528, 14, 14   580K 119M  inception (4d)
            Inception(in_channels=528, out_channels=832,  spec=(256, (160, 320), (32, 128), 128)),      # ->  832, 14, 14   840K 170M  inception (4e)
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                                   # ->  832,  7,  7  (max)
            Inception(in_channels=832, out_channels=832,  spec=(256, (160, 320), (32, 128), 128)),      # ->  832,  7,  7  1072K 54M   inception (5a)
            Inception(in_channels=832, out_channels=1024, spec=(384, (192, 384), (48, 128), 128)),      # -> 1024,  7,  7  1388K 71M   inception (5b)
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                                    # -> 1024, 1, 1
           Flatten(),                                                                                   # -> 1024
           Dropout(0.4),                                                                                #
           Linear(input_size=1024, output_size=n_classes)                                               # -> n_classes(1000)
        )

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
        x = torch.randn(n_samples, 3, 224, 224, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


class DeepPlainCNN(Module):  # used for comparison to ResNet-18

    def __init__(self, n_classes=1000):

        def ConvBlock(n_convs, in_channels, out_channels, downsample=False):
            block = Sequential()
            for i in range(n_convs):  # skips first
                if downsample and i == 0:
                    block.add(Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding='same'))
                else:
                    block.add(Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'))
                block.add(BatchNorm2d(out_channels))
                block.add(ReLU())
            return block

        self.stem = Sequential(                                                                    # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same'),       # ->   64, 112, 112
            BatchNorm2d(64), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                              # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ConvBlock(n_convs=6,   in_channels=64, out_channels=64),                               # ->   64,  56,  56
            ConvBlock(n_convs=8,   in_channels=64, out_channels=128, downsample=True),             # ->  128,  28,  28
            ConvBlock(n_convs=12, in_channels=128, out_channels=256, downsample=True),             # ->  256,  14,  14
            ConvBlock(n_convs=6,  in_channels=256, out_channels=512, downsample=True),             # ->  512,   7,   7
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                               # -> 512, 1, 1
           Flatten(),                                                                              # -> 512
           Linear(input_size=512, output_size=n_classes)                                           # -> n_classes(1000)
        )

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
        x = torch.randn(n_samples, 3, 224, 224, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


# ResNet and other residual networks are moved into "residual_networks.py" module
