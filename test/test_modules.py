import pytest
import torch
from lib.layers import (Linear, Conv2d, Conv2dGroups, MaxPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d, LocalResponseNorm,
                        DotProductAttention, MultiHeadAttention, SparseMultiHeadAttention,
                        PositionalEncoding, RotaryEncoding)
from utils.other import paddings_mask
import einops as ein
from math import sqrt
from matplotlib import pyplot as plt

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
    assert torch.allclose(bn1.running_var.flatten(), bn2.running_var.flatten())
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)

@pytest.mark.parametrize('size',  [1, 2, 5, 10, 99])
def test_batch_norm2d(size):
    x = torch.randn(11, size, 224, 224)
    bn1 = torch.nn.BatchNorm2d(size)
    bn2 = BatchNorm2d(size)
    expected = bn1(x)
    output = bn2.forward(x)
    assert torch.allclose(bn1.running_mean.flatten(), bn2.running_mean.flatten())
    assert torch.allclose(bn1.running_var.flatten(), bn2.running_var.flatten())
    assert torch.allclose(expected, output, rtol=1e-04, atol=1e-06)



def test_dot_product_attention():
    b, q, emb, k, emb_v = 1024, 3, 2, 10, 4
    queries = torch.randn(b, q, emb)
    keys    = torch.randn(b, k, emb)
    values  = torch.randn(b, k, emb_v)
    valid_lens = torch.randint(1, k, (b, q))

    attn = DotProductAttention(dropout=0., scaled=True)
    v1, a1 = attn.forward(queries, keys, values)
    v2 = torch.nn.functional.scaled_dot_product_attention(queries, keys, values)
    assert torch.allclose(v1, v2, rtol=1e-4, atol=1e-6)

    attn_mask = paddings_mask(valid_lens, max_len=k)
    v1, a1 = attn.forward(queries, keys, values, attn_mask)
    v2 = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, ~attn_mask)
    assert torch.allclose(v1, v2, rtol=1e-4, atol=1e-6)



@pytest.mark.parametrize('emb_dim',  [64, 48, 32])
@pytest.mark.parametrize('num_heads',  [1, 4, 8])
@pytest.mark.parametrize('t_source, t_target',  [(15, 15), (10, 12)])
def test_multi_head_attention(emb_dim, num_heads, t_source, t_target):
    b, vocab_size = 10, 1000
    queries = torch.randn(b, t_target, emb_dim)
    keys    = torch.randn(b, t_source, emb_dim)
    values  = torch.randn(b, t_source, emb_dim)
    valid_lens = torch.randint(1, t_source - 1, (b,))
    keys_pad_mask = paddings_mask(valid_lens, max_len=t_source)
    attn_mask = ein.repeat(keys_pad_mask, "b t -> (b h) 1 t", h=num_heads)  # repeat keys_pads along head sub-dimensions

    attention1 = MultiHeadAttention(embed_dim=emb_dim, n_heads=num_heads)
    attention2 = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True, bias=False)

    # use the same parameter values
    wq, wk, wv = attention2.in_proj_weight.data.T.chunk(3, dim=-1)
    attention1.weight_q.data[:] = wq
    attention1.weight_k.data[:] = wk
    attention1.weight_v.data[:] = wv
    attention1.weight_o.data[:] = attention2.out_proj.weight.data.T



    # compute attentions with same x
    if t_source == t_target:
        x = queries
        a1, a_weights1 = attention1.forward(x, x, x, attn_mask=attn_mask), attention1.get_last_attn_weights()
        a2, a_weights2 = attention2.forward(x, x, x, key_padding_mask=keys_pad_mask, average_attn_weights=False)
        assert torch.allclose(a1, a2, rtol=1e-4, atol=1e-6)
        assert torch.allclose(a_weights1, a_weights2, rtol=1e-4, atol=1e-6)

    # compute attentions without keys padding mask
    a1, a_weights1 = attention1.forward(queries, keys, values), attention1.get_last_attn_weights()
    a2, a_weights2 = attention2.forward(queries, keys, values, average_attn_weights=False)
    assert torch.allclose(a1, a1, rtol=1e-4, atol=1e-6)
    assert torch.allclose(a_weights1, a_weights2, rtol=1e-4, atol=1e-6)

    # compute attentions with keys padding mask
    a1, a_weights1 = attention1.forward(queries, keys, values, attn_mask=attn_mask), attention1.get_last_attn_weights()
    a2, a_weights2 = attention2.forward(queries, keys, values, key_padding_mask=keys_pad_mask, average_attn_weights=False)
    assert torch.allclose(a1, a1, rtol=1e-4, atol=1e-6)
    assert torch.allclose(a_weights1, a_weights2, rtol=1e-4, atol=1e-6)

    # compute attentions with causal mask
    causal_mask = torch.triu(torch.ones(t_target, t_source), diagonal=1).bool()
    a1, a_weights1 = attention1.forward(queries, keys, values, attn_mask=causal_mask), attention1.get_last_attn_weights()
    a2, a_weights2 = attention2.forward(queries, keys, values, attn_mask=causal_mask, average_attn_weights=False)
    assert torch.allclose(a1, a1, rtol=1e-4, atol=1e-6)
    assert torch.allclose(a_weights1, a_weights2, rtol=1e-4, atol=1e-6)


    # compute attentions with causal mask and keys padding mask
    a1, a_weights1 = attention1.forward(queries, keys, values, attn_mask=attn_mask | causal_mask), attention1.get_last_attn_weights()
    a2, a_weights2 = attention2.forward(queries, keys, values, attn_mask=causal_mask, key_padding_mask=keys_pad_mask, average_attn_weights=False)
    assert torch.allclose(a1, a1, rtol=1e-4, atol=1e-6)
    assert torch.allclose(a_weights1, a_weights2, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('block_size',  [4, 8])
@pytest.mark.parametrize('seq_len',  [32, 5, 1, 7])
@pytest.mark.parametrize('n_heads',  [1, 4])
def test_sparse_multi_head_attention(block_size, seq_len, n_heads, visualize=True):
    b, t, e = 2, seq_len, 128
    Q = torch.rand(b, t, e)
    K = torch.rand(b, t, e)
    V = torch.rand(b, t, e)

    attention = MultiHeadAttention(e, n_heads, 0)
    sparse_attention = SparseMultiHeadAttention(e, n_heads, 0, block_size)

    # use the same parameter values
    sparse_attention.weight_q.data[:] = attention.weight_q.data.clone()
    sparse_attention.weight_k.data[:] = attention.weight_k.data.clone()
    sparse_attention.weight_v.data[:] = attention.weight_v.data.clone()
    sparse_attention.weight_o.data[:] = attention.weight_o.data.clone()

    Y2 = sparse_attention.forward(Q, K, V)
    A2 = sparse_attention.get_last_attn_weights()

    # simulate the sparsed attention with two heads for each pattern
    attn_mask_diag = sparse_attention.attn_local.get_attn_mask(t)
    attn_mask_col = sparse_attention.attn_global.get_attn_mask(t)
    Y1_diag = attention.forward(Q, K, V, attn_mask=attn_mask_diag)
    A1_diag = attention.get_last_attn_weights()
    Y1_col = attention.forward(Q, K, V, attn_mask=attn_mask_col)
    A1_col = attention.get_last_attn_weights()
    Y1_col[:, :block_size - 1] = 0     # these are expected to be -inf
    A1_col[:, :, :block_size - 1] = 0  # these are expected to be -inf

    Y1 = Y1_diag + Y1_col
    A1 = A1_diag + A1_col

    if visualize:
        visualize_attn(A1[0].view(n_heads * t, t).detach().cpu(), A2[0].view(n_heads * t, t).detach().cpu(), f'{seq_len=}, {block_size=}, {n_heads=}')
    assert torch.allclose(A1, A2)
    assert torch.allclose(Y1, Y2, rtol=1e-04, atol=1e-06)


def visualize_attn(attn_expected, attn_actual, title=''):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.matshow(attn_expected, vmin=0)
    ax1.set_title('Expected')
    ax2.matshow(attn_actual, vmin=0)
    ax2.set_title('Actual')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def test_positional_encodings_frequency():
    t, d = 50, 256
    sin_enc = PositionalEncoding.compute_encodings(d, t, mixed=True)
    rot_enc = RotaryEncoding.compute_encodings(d, t)
    rot_enc_real = torch.view_as_real(rot_enc).view(t, d)
    assert torch.allclose(sin_enc, rot_enc_real)


@pytest.mark.parametrize('t',  [1, 2, 50])
@pytest.mark.parametrize('d',  [2, 8, 64])
@pytest.mark.parametrize('base_theta',  [10_000, 500_000])
def test_positional_rotary_encodings(t, d, base_theta, batch_size=3):
    x = torch.randn(batch_size, t, d)

    encoder = RotaryEncoding(d, t, base_freq_theta=base_theta)
    x_rotated = encoder.forward(x, clockwise=False)
    x_restored = encoder.forward(x_rotated, clockwise=True)

    assert torch.allclose(x, x_rotated, rtol=1e-04, atol=1e-06) == (t == 1)  # expected no rotation only for t=1
    assert torch.allclose(x, x_restored, rtol=1e-04, atol=1e-06)



