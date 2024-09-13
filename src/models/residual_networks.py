import torch

from lib.layers import Module, Sequential, Linear, Conv2d, AvgPool2d, MaxPool2d, BatchNorm2d, ReLU, Flatten, ModuleList, ConvTranspose2d, PositionalEncoding
from models.blocks.convolutional_blocks import ResBlock, ResBottleneckBlock, ResNeXtBlock, DenseBlock, DenseTransition
from preprocessing.integer import one_hot


class ResNet34(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_classes=1000, attention=False):

        self.stem = Sequential(                                                                # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', bias=False),   # ->   64, 112, 112
            BatchNorm2d(64), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                          # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ResBlock(in_channels=64, out_channels=64, attention=attention),                    # ->   64,  56,  56 (stride /2)
            ResBlock(in_channels=64, out_channels=64, attention=attention),                    # ->   64,  56,  56
            ResBlock(in_channels=64, out_channels=64, attention=attention),                    # ->   64,  56,  56

            ResBlock(in_channels=64, out_channels=128, attention=attention, stride=2),         # ->  128,  28,  28 (stride /2)
            ResBlock(in_channels=128, out_channels=128, attention=attention),                  # ->  128,  28,  28
            ResBlock(in_channels=128, out_channels=128, attention=attention),                  # ->  128,  28,  28
            ResBlock(in_channels=128, out_channels=128, attention=attention),                  # ->  128,  28,  28

            ResBlock(in_channels=128, out_channels=256, attention=attention, stride=2),        # ->  256,  14,  14 (stride /2)
            ResBlock(in_channels=256, out_channels=256, attention=attention),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention),                  # ->  256,  14,  14
            ResBlock(in_channels=256, out_channels=256, attention=attention),                  # ->  256,  14,  14

            ResBlock(in_channels=256, out_channels=512, attention=attention, stride=2),        # ->  512,   7,   7 (stride /2)
            ResBlock(in_channels=512, out_channels=512, attention=attention),                  # ->  512,   7,   7
            ResBlock(in_channels=512, out_channels=512, attention=attention),                  # ->  512,   7,   7
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                           # -> 512, 1, 1
           Flatten(),                                                                          # -> 512
           Linear(input_size=512, output_size=n_classes)                                       # -> n_classes(1000)
        )

    def forward(self, x, verbose=False):
        N, C, H, W = x.shape
        assert (C, H, W) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, H, W)}'

        x = self.stem.forward(x, verbose)
        x = self.body.forward(x, verbose)
        x = self.head.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 224, 224, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


class ResNet50(Module):
    """
    Paper: Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_classes=1000, attention=False):

        self.stem = Sequential(                                                                                             # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', bias=False),                    # ->   64, 112, 112
            BatchNorm2d(64), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                                                       # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ResBottleneckBlock(in_channels=64,  mid_channels=64, out_channels=256, attention=attention),                 # 64 ->  [64] -> 256,  56,  56 (stride /2)
            ResBottleneckBlock(in_channels=256, mid_channels=64, out_channels=256, attention=attention),                # 256 ->  [64] -> 256,  56,  56
            ResBottleneckBlock(in_channels=256, mid_channels=64, out_channels=256, attention=attention),                # 256 ->  [64] -> 256,  56,  56

            ResBottleneckBlock(in_channels=256, mid_channels=128, out_channels=512, attention=attention, stride=2),     # 256 -> [128] -> 512,  28,  28 (stride /2)
            ResBottleneckBlock(in_channels=512, mid_channels=128, out_channels=512, attention=attention),               # 512 -> [128] -> 512,  28,  28
            ResBottleneckBlock(in_channels=512, mid_channels=128, out_channels=512, attention=attention),               # 512 -> [128] -> 512,  28,  28
            ResBottleneckBlock(in_channels=512, mid_channels=128, out_channels=512, attention=attention),               # 512 -> [128] -> 512,  28,  28

            ResBottleneckBlock(in_channels=512,  mid_channels=256, out_channels=1024, attention=attention, stride=2),   # 512 -> [256] -> 1024, 28,  28 (stride /2)
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention),            # 1024 -> [256] -> 1024, 28,  28
            ResBottleneckBlock(in_channels=1024, mid_channels=256, out_channels=1024, attention=attention),            # 1024 -> [256] -> 1024, 28,  28

            ResBottleneckBlock(in_channels=1024, mid_channels=512, out_channels=2048, attention=attention, stride=2),  # 1024 -> [512] -> 2048,  7,   7 (stride /2)
            ResBottleneckBlock(in_channels=2048, mid_channels=512, out_channels=2048, attention=attention),            # 2048 -> [512] -> 2048,  7,   7
            ResBottleneckBlock(in_channels=2048, mid_channels=512, out_channels=2048, attention=attention),            # 2048 -> [512] -> 2048,  7,   7
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                                                   # -> 2048, 1, 1
           Flatten(),                                                                                                  # -> 2048
           Linear(input_size=2048, output_size=n_classes)                                                              # -> n_classes(1000)
        )

    def forward(self, x, verbose=False):
        N, C, H, W = x.shape
        assert (C, H, W) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, H, W)}'

        x = self.stem.forward(x, verbose)
        x = self.body.forward(x, verbose)
        x = self.head.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 224, 224, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


class ResNeXt50(Module):  # same computations/params as ResNet-50, but more channels and better accuracy
    """
    Paper: Aggregated Residual Transformations for Deep Neural Networks
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf
    """

    def __init__(self, n_classes=1000, attention=False):

        self.stem = Sequential(                                                                                               # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', bias=False),                      # ->   64, 112, 112
            BatchNorm2d(64), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                                                         # ->   64,  56,  56 (max)
        )

        self.body = Sequential(
            ResNeXtBlock(in_channels=64,  mid_channels=128, out_channels=256, groups=32, attention=attention),                 # 64 ->  [128] -> 256,  56,  56 (stride /2)
            ResNeXtBlock(in_channels=256, mid_channels=128, out_channels=256, groups=32, attention=attention),                # 256 ->  [128] -> 256,  56,  56
            ResNeXtBlock(in_channels=256, mid_channels=128, out_channels=256, groups=32, attention=attention),                # 256 ->  [128] -> 256,  56,  56

            ResNeXtBlock(in_channels=256, mid_channels=256, out_channels=512, groups=32, attention=attention, stride=2),      # 256 ->  [256] -> 512,  28,  28 (stride /2)
            ResNeXtBlock(in_channels=512, mid_channels=256, out_channels=512, groups=32, attention=attention),                # 512 ->  [256] -> 512,  28,  28
            ResNeXtBlock(in_channels=512, mid_channels=256, out_channels=512, groups=32, attention=attention),                # 512 ->  [256] -> 512,  28,  28
            ResNeXtBlock(in_channels=512, mid_channels=256, out_channels=512, groups=32, attention=attention),                # 512 ->  [256] -> 512,  28,  28

            ResNeXtBlock(in_channels=512,  mid_channels=512, out_channels=1024, groups=32, attention=attention, stride=2),    # 512 ->  [512] -> 1024, 28,  28 (stride /2)
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention),             # 1024 ->  [512] -> 1024, 28,  28
            ResNeXtBlock(in_channels=1024, mid_channels=512, out_channels=1024, groups=32, attention=attention),             # 1024 ->  [512] -> 1024, 28,  28

            ResNeXtBlock(in_channels=1024, mid_channels=1024, out_channels=2048, groups=32, attention=attention, stride=2),  # 1024 -> [1024] -> 2048,  7,   7 (stride /2)
            ResNeXtBlock(in_channels=2048, mid_channels=1024, out_channels=2048, groups=32, attention=attention),            # 2048 -> [1024] -> 2048,  7,   7
            ResNeXtBlock(in_channels=2048, mid_channels=1024, out_channels=2048, groups=32, attention=attention),            # 2048 -> [1024] -> 2048,  7,   7
        )
        self.head = Sequential(
           AvgPool2d(kernel_size=7),                                                                             # -> 2048, 1, 1
           Flatten(),                                                                                            # -> 2048
           Linear(input_size=2048, output_size=n_classes)                                         # -> n_classes(1000)
        )

    def forward(self, x, verbose=False):
        N, C, H, W = x.shape
        assert (C, H, W) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, H, W)}'

        x = self.stem.forward(x, verbose)
        x = self.body.forward(x, verbose)
        x = self.head.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 224, 224, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)


class SEResNet50(ResNet50):
    """
    Paper: Squeeze-and-Excitation Networks
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, n_classes=1000):
        super().__init__(n_classes, attention=True)


class SEResNeXt50(ResNeXt50):
    """
    Paper: Squeeze-and-Excitation Networks
    https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, n_classes=1000):
        super().__init__(n_classes, attention=True)


class DenseNet121(Module):  # DenseNet-BC (bottleneck + compression)
    """
    Paper: Densely Connected Convolutional Networks
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    """

    def __init__(self, n_classes=1000, dropout=.2):
        self.stem = Sequential(                                                                # in:   3, 224, 224
            Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding='same', bias=False),   # ->   64, 112, 112
            BatchNorm2d(64), ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=(0, 1, 0, 1)),                          # ->   64,  56,  56 (max)
        )
        self.body = Sequential(
            DenseBlock(in_channels=64, growth_rate=32, n_convs=6, dropout=dropout),            # ->  256,  56,  56  [64 + 32*6]
            DenseTransition(in_channels=256, out_channels=128, downsample_by=2),               # ->  256,  28,  28  (avg)
            DenseBlock(in_channels=128, growth_rate=32, n_convs=12, dropout=dropout),          # ->  512,  28,  28  [128 + 32*12]
            DenseTransition(in_channels=512, out_channels=256, downsample_by=2),               # ->  256,  14,  14  (avg)
            DenseBlock(in_channels=256, growth_rate=32, n_convs=24, dropout=dropout),          # -> 1024,  14,  14  [256 + 32*24]
            DenseTransition(in_channels=1024, out_channels=512, downsample_by=2),              # ->  512,   7,   7  (avg)
            DenseBlock(in_channels=512, growth_rate=32, n_convs=16, dropout=dropout),          # -> 1024,   7,   7  [512 + 32*16]
        )
        self.head = Sequential(
            BatchNorm2d(1024), ReLU(),
            AvgPool2d(kernel_size=7),                                                          # ->  1024, 1, 1
            Flatten(),                                                                         # ->  1024
            Linear(input_size=1024, output_size=n_classes)                                     # ->  n_classes(1000)
        )

    def forward(self, x, verbose=False):
        N, C, H, W = x.shape
        assert (C, H, W) == (3, 224, 224), f'Expected input shape {(3, 224, 224)} but got {(C, H, W)}'

        x = self.stem.forward(x, verbose)
        x = self.body.forward(x, verbose)
        x = self.head.forward(x, verbose)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        x = torch.randn(n_samples, 3, 224, 224, device=self.device_of_first_parameter())
        return self.forward(x, verbose=True)




class Down(Module):
    """ignore docs"""
    def __init__(self, in_channels, out_channels, time_channels):
        self.time_emb = Sequential(Linear(time_channels, in_channels), ReLU())
        self.conv = ResBlock(in_channels, out_channels, attention=True, mem_optimized=True)
        self.downscale = Conv2d(out_channels, out_channels, kernel_size=2, stride=2)  # the skip connections between encoder/decoder should provide the lost information
        # self.downscale = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, t):
        B, C, H, W = x.shape
        x = x + self.time_emb(t).view(B, C, 1, 1)    # note the time embeddings are added before the residuals (in contrast to the paper)
        x_keep = self.conv(x)                        # B, C, H, W  -> B, 2C, H, W
        x = self.downscale(x_keep)                   # B, 2C, H, W -> B, 2C, H/2, W/2
        return x, x_keep


class Middle(Module):
    """ignore docs"""
    def __init__(self, in_channels, out_channels, time_channels):
        self.time_emb = Sequential(Linear(time_channels, in_channels), ReLU())
        self.conv = ResBlock(in_channels, out_channels, attention=True, mem_optimized=True)

    def forward(self, x, t):
        B, C, H, W = x.shape
        x = x + self.time_emb(t).view(B, C, 1, 1)    # note the time embeddings are added before the residuals (in contrast to the paper)
        x = self.conv(x)
        return x


class Up(Module):
    """ignore docs"""
    def __init__(self, in_channels, in_skip_channels, out_channels, time_channels):
        self.upscale = ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2, mem_optimized=True)
        self.time_emb = Sequential(Linear(time_channels, in_channels//2), ReLU())
        self.conv = ResBlock(in_channels//2 + in_skip_channels, out_channels, attention=True, mem_optimized=True)

    def forward(self, x, x_skip, t):
        x = self.upscale(x)                                     # B, 2C, 2H, 2W -> B, C, H, W
        B, C, _, _ = x.shape
        x = x + self.time_emb(t).view(B, C, 1, 1)      # note the time embeddings are added before the residuals (in contrast to the paper)
        x = self.conv(torch.cat((x, x_skip), dim=1))    # B, [C, C_skip], H, W -> B, C, H, W
        return x


class UNet_DDPM(Module):
    """
    Very loose implementation of the DDPM U-net with residuals, channel-wise attention and timestep embeddings
    """

    def __init__(self, img_sizes=(1, 32, 32), context_features=10, max_timesteps=100):
        self.img_sizes, self.context_features, self.max_timesteps = img_sizes, context_features, max_timesteps
        C, H, W = img_sizes
        assert H == W and H % 2**4 == 0, f'The image size must be divisible by 2^4 (for proper down and up scaling): but got {H}x{W}'

        self.time_emb = PositionalEncoding(128, max_seq_len=max_timesteps)
        self.context_emb = Sequential(
            Linear(context_features, 256),
            ReLU(),
            Linear(256, 128),
        )
                                                                                                               # in:   1, 32, 32
        self.proj = Conv2d(in_channels=C, out_channels=64, kernel_size=1, padding='same', mem_optimized=True)  # ->   64, 32, 32
        self.down = ModuleList([
            Down(in_channels=64,   out_channels=64,  time_channels=256),                                       # ->   64, 32, 32
            Down(in_channels=64,   out_channels=128, time_channels=256),                                       # ->  128, 16, 16
            Down(in_channels=128,  out_channels=256, time_channels=256),                                       # ->  256,  8,  8
            Down(in_channels=256,  out_channels=512, time_channels=256),                                       # ->  512,  4,  4
        ])
        self.middle = Middle(in_channels=512, out_channels=1024, time_channels=256)                            # -> 1024,  2,  2
        self.up = ModuleList([
            Up(in_channels=1024, in_skip_channels=512, out_channels=512, time_channels=256),                   # ->  512,  4,  4
            Up(in_channels=512,  in_skip_channels=256, out_channels=256, time_channels=256),                   # ->  256,  8,  8
            Up(in_channels=256,  in_skip_channels=128, out_channels=128, time_channels=256),                   # ->  128, 16, 16
            Up(in_channels=128,  in_skip_channels=64,  out_channels=64,  time_channels=256),                   # ->   64, 32, 32
        ])
        self.out = Conv2d(in_channels=64, out_channels=C, kernel_size=1, padding='same', mem_optimized=True)   # ->    1, 32, 32


    def forward(self, x, t, context):
        # Timestep & context embeddings
        t = self.time_emb.fixed_embeddings[t]   # B, E
        c = self.context_emb(context)           # B, E
        tc = torch.cat((t, c), dim=1)           # B, 2E

        # Encoder ---------------------------------------
        B, C, H, W = x.shape
        x = self.proj(x)
        x_skip = []   # store dense skip connections
        for down in self.down:
            x, x_keep = down(x, tc)
            x_skip.append(x_keep)

        # Code ---------------------------------------
        x = self.middle(x, tc)

        # Decoder ---------------------------------------
        for up in self.up:
            x = up(x, x_skip.pop(), tc)
        x = self.out(x)

        return x

    @torch.no_grad()
    def test(self, n_samples=1):
        B, (C, H, W) = n_samples, self.img_sizes
        device = self.device_of_first_parameter()

        x = torch.randn(B, C, H, W).to(device)
        t = torch.randint(1, self.max_timesteps+1, (B, )).to(device)
        context = one_hot(torch.randint(self.context_features, (B,)), num_classes=self.context_features).to(device)
        return self.forward(x.to(device), t.to(device), context.to(device))
