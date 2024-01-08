import pytest
import torch
from models.residual_networks import ResNet34, ResNet50, ResNeXt50
from utils.rng import seed_global

@torch.no_grad()
def test_ResNet34():
    net = ResNet34(n_classes=1000, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)

@torch.no_grad()
def test_ResNet50():
    net = ResNet50(n_classes=1000, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)

@torch.no_grad()
def test_ResNeXt50():
    net = ResNeXt50(n_classes=1000, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


@torch.no_grad()
def test_SEResNet50():
    net = ResNet50(n_classes=1000, attention=True, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


@torch.no_grad()
def test_SEResNeXt50():
    net = ResNeXt50(n_classes=1000, attention=True, device='cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


