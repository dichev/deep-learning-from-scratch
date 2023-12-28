import pytest
import torch
from models.convolutional_networks import SimpleCNN, LeNet5, AlexNet, NetworkInNetwork, VGG16
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

