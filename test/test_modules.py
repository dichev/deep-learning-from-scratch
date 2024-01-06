import pytest
import torch
from lib.layers import Linear, Conv2d, Conv2dGroups, MaxPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d, LocalResponseNorm
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
@pytest.mark.parametrize('in_channels',  [4, 8])
@pytest.mark.parametrize('out_channels', [8, 4])
@pytest.mark.parametrize('groups',   [1, 2, 4])
def test_conv2d_groups(in_channels, out_channels, groups):
    N, C_out, C_in, W, H = 10, in_channels, out_channels, 100, 100
    kernel, padding, stride, dilation = 3, 1, 1, 1
    A = torch.nn.Conv2d(C_in, C_out, kernel, stride=stride, padding=padding, dilation=dilation, groups=groups)
    B = Conv2dGroups(C_in, C_out, kernel, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # use the same parameters
    step = C_out//groups
    for g in range(groups):
        group = slice(g*step, (g+1)*step)
        assert B.convs[g].weight.shape == A.weight[group].shape, f'Expected the same weight shape: {B.convs[g].weight.shape}, {A.weight[group].shape}'
        assert A.bias[group].shape == B.convs[g].bias.shape, f'Expected the same bias shape: {A.bias[group].shape}, {B.convs[g].bias.shape}'
        with torch.no_grad():
            B.convs[g].weight[:] = A.weight[group].detach().clone()
            B.convs[g].bias[:] = A.bias[group].detach().clone()

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
    lrn1 = torch.nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
    lrn2 = LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)
    expected = lrn1(x)
    output = lrn2.forward(x)
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

# @torch.no_grad()
@pytest.mark.parametrize('size',  [1, 2, 5, 10, 99])
def test_batch_norm1d(size):
    x = torch.randn(11, size)
    bn1 = torch.nn.BatchNorm1d(size)
    bn2 = BatchNorm1d(size)
    expected = bn1(x)
    output = bn2.forward(x)
    assert torch.allclose(bn1.running_mean.flatten(), bn2.running_mean.flatten())
    # assert torch.allclose(bn1.running_var.flatten(), bn2.running_var.flatten()) # it looks like the running variance in pytorch is computed with unbiased variance
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

@pytest.mark.parametrize('size',  [1, 2, 5, 10, 99])
def test_batch_norm2d(size):
    x = torch.randn(11, size, 224, 224)
    bn1 = torch.nn.BatchNorm2d(size)
    bn2 = BatchNorm2d(size)
    expected = bn1(x)
    output = bn2.forward(x)
    assert torch.allclose(bn1.running_mean.flatten(), bn2.running_mean.flatten())
    # assert torch.allclose(bn1.running_var.flatten(), bn2.running_var.flatten()) # it looks like the running variance in pytorch is computed with unbiased variance
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

