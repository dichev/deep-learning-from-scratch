import pytest
import torch
from lib.functions.metrics import BLEU
from lib.functions.activations import softmax, log_softmax
from utils.other import paddings_mask
from torchtext.data.metrics import bleu_score
import math


@pytest.mark.parametrize('z_shape',  [(2, 8), (2, 4, 6), (1024, 16, 32)])
def test_masked_softmax(z_shape):
    max_len = z_shape[-1]
    z = torch.randn(*z_shape)
    lengths = torch.randint(1, max_len, z_shape[:-1])

    mask = paddings_mask(lengths, max_len)
    p1 = softmax(z)
    p2 = softmax(z, ignore_mask=mask)
    p3 = log_softmax(z, ignore_mask=mask).exp()
    valid_probs = torch.ones(z.shape[:-1])

    assert torch.allclose(p1.sum(dim=-1), valid_probs)
    assert torch.allclose(p2.sum(dim=-1), valid_probs) and torch.all(p2[mask] == 0.)
    assert torch.allclose(p3.sum(dim=-1), valid_probs) and torch.all(p3[mask] == 0.)
    assert torch.allclose(p2, p3)


@pytest.mark.parametrize('max_n', [2, 3, 4])
def test_BLEU(max_n):
    n = torch.arange(max_n) + 1
    weights = (1 / 2 ** n)

    targets = [
        'Translate me correctly, please',
        'Something for translation is written here',
        'This is another test sentence for evaluation',
        'The quick brown fox jumps over the lazy dog',
        'Complexity adds depth to simple statements',
        'New test cases improve coverage',
        'Localization vs. globalization is an ongoing debate',
        'Consistency is key for quality translations',
    ]

    translations = [
        'Translate me correctly',
        'Something for translation was written here',
        'This another test sentence is for evaluation',
        'The quick brown fox leaped over a lazy dog',
        'Complexity gives depth to simplistic statements',
        'New test scenarios increase coverage',
        'The debate continues between localization and globalization',
        'Consistent is key for translations quality ',
    ]

    for y, y_hat in zip(targets, translations):
        y, y_hat = y.split(), y_hat.split()
        expected = bleu_score([y_hat], [[y]], max_n, weights.tolist())
        actual = BLEU(y_hat, y, max_n)
        # print(expected, actual)
        assert math.isclose(expected, actual, rel_tol=1e-04, abs_tol=1e-06)
