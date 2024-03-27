import pytest
import torch
from lib.functions.metrics import BLEU
from torchtext.data.metrics import bleu_score
import math


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
