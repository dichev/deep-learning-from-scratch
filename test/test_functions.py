import pytest
import torch
from lib.functions.metrics import BLEU
from lib.functions.activations import softmax, log_softmax, relu, silu, gelu, swish
from utils.other import paddings_mask
from utils import rng
from utils.images import window_partition, window_reverse
from torcheval.metrics.functional import bleu_score
import math, random, numpy as np


@pytest.mark.parametrize('batch',  [1, 2, 5, 256])
@pytest.mark.parametrize('dim',  [1, 2, 5, 256])
def test_smooth_relu(batch, dim):
    x = torch.randn(batch, dim)
    assert torch.allclose(silu(x), swish(x, 1))
    assert torch.allclose(gelu(x), swish(x, 1.702), rtol=1e-1, atol=1e-1)  # approx
    assert torch.allclose(relu(x), swish(x, torch.inf))


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
        'Translate me correctly, now',
        'Something for translation was written here',
        'This another test sentence is for evaluation',
        'The quick brown fox leaped over a lazy dog',
        'Complexity gives depth to simplistic statements',
        'New test scenarios increase coverage',
        'The debate continues between localization and globalization',
        'Consistent is key for translations quality ',
    ]

    for y, y_hat in zip(targets, translations):
        expected = bleu_score([y_hat], [[y]], max_n, weights)
        actual = BLEU(y_hat.split(), y.split(), max_n)
        assert math.isclose(expected, actual, rel_tol=1e-04, abs_tol=1e-06)


def test_rng_seed():
    rng.seed_global(1)
    rng_numbers_A = generate_random_numbers()

    rng.seed_global(1)
    rng_numbers_B = generate_random_numbers()

    assert rng_numbers_A['random'] == rng_numbers_B['random']
    assert rng_numbers_A['numpy'] == rng_numbers_B['numpy']
    assert rng_numbers_A['torch'] == rng_numbers_B['torch']
    assert rng_numbers_A['cuda'] == rng_numbers_B['cuda']


def test_rng_states():
    init_state = rng.get_rng_states()
    rng_numbers_A = generate_random_numbers()

    rng.set_rng_states(init_state)
    rng_numbers_B = generate_random_numbers()

    assert rng_numbers_A['random'] == rng_numbers_B['random']
    assert rng_numbers_A['numpy'] == rng_numbers_B['numpy']
    assert rng_numbers_A['torch'] == rng_numbers_B['torch']
    assert rng_numbers_A['cuda'] == rng_numbers_B['cuda']


def generate_random_numbers(n=10):
    return {
        'random': [random.random() for _ in range(n)],
        'numpy': np.random.randn(n).tolist(),
        'torch': torch.randn(n).tolist(),
        'cuda': torch.cuda.FloatTensor(n).normal_().tolist() if torch.cuda.is_available() else None
    }


def test_windows():
    B, C, H, W = 2, 3, 8, 12
    win_size = 4

    a = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
         [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
         [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
         [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
         [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
         [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
         [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
         [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]]
    )
    b = a.view(1, 1, 8, 12).expand(B, C, H, W)
    b_win = window_partition(b, win_size)   # B, C, H, W   ->  B, N, C, H/win_size, W/win_size

    # assert all tensor elements across the same dim are equal
    assert torch.all(b_win[0, :, 0].sum(dim=[-2, -1]) == torch.tensor([0, 1, 2, 3, 4, 5]) * win_size * win_size)

    b_reverse = window_reverse(b_win, window_size=win_size, height=H, width=W)
    assert torch.all(b == b_reverse)

