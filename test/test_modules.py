import pytest
import torch
from lib.layers import Linear, Conv2d, MaxPool2d, AvgPool2d, LocalResponseNorm
from utils.rng import seed_global

@torch.no_grad()
def test_linear():
    seed_global(1)
    x = torch.tensor([[1., 2.], [3., 4.]])
    y = Linear(2, 3).forward(x)
    expected = torch.tensor([[1.9040, -0.6369, -0.2706], [4.4693, -1.0069, -0.4795]])
    assert torch.allclose(y, expected, rtol=1e-4, atol=1e-4)

@torch.no_grad()
@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('stride',   [1, 2, 3])
@pytest.mark.parametrize('padding',  [0, 1, 2, 3])
@pytest.mark.parametrize('kernel',   [1, 3, 5, 7])
def test_conv2d(kernel, padding, stride, dilation):
    N, C_out, C_in, W, H = 10, 4, 3, 100, 100
    A = torch.nn.Conv2d(C_in, C_out, kernel, stride=stride, padding=padding, dilation=dilation)
    B = Conv2d(C_in, C_out, kernel, stride=stride, padding=padding, dilation=dilation)

    # use the same parameters
    assert B.weight.shape == A.weight.shape, f'Expected the same weight shape: {B.weight.shape}, {A.weight.shape}'
    assert A.bias.shape == B.bias.shape, f'Expected the same bias shape: {A.bias.shape}, {B.bias.shape}'
    with torch.no_grad():
        B.weight[:] = A.weight.detach().clone()
        B.bias[:] = A.bias.detach().clone()

    # compare the convolutions
    input = torch.randn(N, C_in, W, H)
    expected = A(input)
    output = B.forward(input)
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

@torch.no_grad()
@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('stride',   [1, 2, 3])
@pytest.mark.parametrize('padding, kernel',  [(0, 1), (0, 3), (0, 5), (1, 3), (2, 5)])
def test_max_pool2d(kernel, padding, stride, dilation):
    N, C, W, H = 10, 3, 100, 100
    A = torch.nn.MaxPool2d(kernel, stride=stride, padding=padding, dilation=dilation)
    B = MaxPool2d(kernel, stride=stride, padding=padding, dilation=dilation)

    input = torch.randn(N, C, W, H)
    expected = A(input)
    output = B.forward(input)
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

@torch.no_grad()
@pytest.mark.parametrize('stride',   [1, 2, 3])
@pytest.mark.parametrize('padding, kernel',  [(0, 1), (0, 3), (0, 5), (1, 3), (2, 5)])
def test_avg_pool2d(kernel, padding, stride):
    N, C, W, H = 10, 3, 100, 100
    A = torch.nn.AvgPool2d(kernel, stride=stride, padding=padding)
    B = AvgPool2d(kernel, stride=stride, padding=padding)

    input = torch.randn(N, C, W, H)
    expected = A(input)
    output = B.forward(input)
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

@torch.no_grad()
@pytest.mark.parametrize('size, alpha, beta, k',  [(5, 5*1e-4, .75, 2.), (3, 1e-2, .75, .5), (7, 1e-1, .15, .1)])
def test_avg_pool2d(size, alpha, beta, k):
    x = torch.randn(11, 32, 10, 10)
    from torch import nn
    lrn1 = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
    lrn2 = LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
    expected = lrn1(x)
    output = lrn2.forward(x)
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

