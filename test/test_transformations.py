import pytest
import torch
import torch.nn.functional as F
from utils import images as I


def test_affine_grid_identity(batch_size=2, channels=3, height=32, width=64):
    theta = torch.tensor([[
        [1., 0., 0.],
        [0., 1., 0.],
    ]]).expand(batch_size, 2, 3)
    img = torch.randn(batch_size, channels, height, width)

    actual = I.affine_grid(theta, img.size())
    expected = F.affine_grid(theta, img.size(), align_corners=True)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('batch_size', [1, 5, 10])
@pytest.mark.parametrize('channels',   [1, 2, 3])
@pytest.mark.parametrize('height',     [32, 64])
def test_affine_grid(batch_size, channels, height, width=64):
    theta = torch.randn(batch_size, 2, 3)
    img = torch.randn(batch_size, channels, height, width)

    actual = I.affine_grid(theta, img.size())
    expected = F.affine_grid(theta, img.size(), align_corners=True)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('batch_size', [1, 5, 10])
@pytest.mark.parametrize('channels',   [1, 2, 3])
@pytest.mark.parametrize('height',     [32, 64])
def test_transform_image(batch_size, channels, height, width=64):
    theta = torch.randn(batch_size, 2, 3)
    img = torch.randn(batch_size, channels, height, width)
    grid = I.affine_grid(theta, img.size())

    actual = I.transform_image(img, grid, interpolation='bilinear')
    expected = F.grid_sample(img, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)


