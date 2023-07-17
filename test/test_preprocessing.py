import pytest
import torch
import pandas as pd

from preprocessing.floats import normalizeMinMax
from preprocessing.integer import index_encoder

def test_normalizeMinMax():
    # Test with a tensor of positive integers
    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.all(normalizeMinMax(x) == y)

    # Test with a tensor of negative integers
    x = torch.tensor([-5, -4, -3, -2, -1])
    y = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.all(normalizeMinMax(x) == y)

    # Test with a mix of positive and negative integers
    x = torch.tensor([-5, 0, 5])
    y = torch.tensor([0.0, 0.5, 1.0])
    assert torch.all(normalizeMinMax(x) == y)

    # Test with a tensor of zeros and non-zero number
    a = torch.Tensor([[0., 0., 0.], [5., 5., 5.]])
    b = torch.Tensor([[0., 0., 0.], [0., 0., 0.]])
    assert torch.all(normalizeMinMax(a) == b)

    # Test with a tensor on adifferent dim value
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert torch.all(normalizeMinMax(x, dim=0) == y)

