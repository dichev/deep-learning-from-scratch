import pytest
import torch
from models.convolutional_networks import SimpleCNN, LeNet5, AlexNet, NetworkInNetwork, VGG16, Inception, GoogLeNet
from utils.rng import seed_global

@torch.no_grad()
def test_SimpleCNN():
    net = SimpleCNN(device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 10)


@torch.no_grad()
def test_LeNet5():
    net = LeNet5(device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 10)

@torch.no_grad()
def test_AlexNet():
    net = AlexNet(n_classes=1000, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)

@torch.no_grad()
def test_NetworkInNetwork():
    net = NetworkInNetwork(n_classes=1000, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)

@torch.no_grad()
def test_VGG16():
    net = VGG16(n_classes=1000, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


@torch.no_grad()
def test_Inception():
    N, W, H = 2, 50, 50
    inception = Inception(in_channels=3, out_channels=512, spec=(192, (96, 208), (16, 48), 64))
    data = torch.randn(N, 3, W, H)
    out = inception.forward(data)
    assert out.shape == (N, 512, W, H)

@torch.no_grad()
def test_GoogLeNet():
    net = GoogLeNet(n_classes=1000, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


