import pytest
import torch
from models.residual_networks import ResNet34, ResNet50, ResNeXt50, DenseNet121
from utils.rng import seed_global

@torch.no_grad()
def test_ResNet34():
    net = ResNet34(n_classes=1000).to('cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)

@torch.no_grad()
def test_ResNet50():
    net = ResNet50(n_classes=1000).to('cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)

@torch.no_grad()
def test_ResNeXt50():
    net = ResNeXt50(n_classes=1000).to('cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


@torch.no_grad()
def test_SEResNet50():
    net = ResNet50(n_classes=1000, attention=True).to('cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


@torch.no_grad()
def test_SEResNeXt50():
    net = ResNeXt50(n_classes=1000, attention=True).to('cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


@torch.no_grad()
def test_DenseNet121():
    net = DenseNet121(n_classes=1000, dropout=.2).to('cuda')
    out = net.test(n_samples=9)
    assert out.shape == (9, 1000)


