import pytest
import torch
from models.transformer_networks import LLaMA1, LLaMA2
from models.visual_transformers import SwinTransformerBlock
from lib.layers import MultiHeadAttention, GroupedQueryAttention
from utils.rng import seed_global
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

@pytest.mark.parametrize('batch_size', [1, 16])
@pytest.mark.parametrize('prompt_size', [1, 5, 100])
def test_LLAMA_caching(prompt_size, batch_size):
    vocab_size, context_size = 500, 200
    model = LLaMA1(vocab_size, context_size, embed_size=256, hidden_size=4*256, n_layers=4, attn_heads=4).to('cuda')
    prompt = torch.randint(vocab_size, (batch_size, prompt_size)).to('cuda')

    seed_global(1)
    tokens_cached = model.generate(prompt, max_tokens=context_size - prompt_size, use_cache=True, from_topk=20) # note on higher topk there could be difference due to floating point precision differences
    seed_global(1)
    tokens_not_cached = model.generate(prompt, max_tokens=context_size - prompt_size, use_cache=False, from_topk=20)
    assert torch.allclose(tokens_cached, tokens_not_cached)


@pytest.mark.parametrize('emb_size', [64, 128])
@pytest.mark.parametrize('groups', [1, 2, 8, 16])
@pytest.mark.parametrize('n_layers', [1, 2])
def test_LLAMA_with_gqa_for_parameter_size_mismatch(emb_size, groups, n_layers):
    model_mha = LLaMA2(vocab_size=1000, context_size=100, embed_size=emb_size, hidden_size=emb_size * 4, n_layers=n_layers, attn_heads=16, attn_kv_groups=0)
    model_gqa = LLaMA2(vocab_size=1000, context_size=100, embed_size=emb_size, hidden_size=emb_size * 4, n_layers=n_layers, attn_heads=16, attn_kv_groups=groups)
    assert isinstance(model_mha.transformers[0].attn, MultiHeadAttention)
    assert isinstance(model_gqa.transformers[0].attn, GroupedQueryAttention)
    assert abs(model_mha.count_params() - model_gqa.count_params()) <= emb_size * n_layers  # small difference is expected due to rounding in ff layers


def test_SwinTrasnformer_shifted_window_attn_masks(visualize=True):
    swin = SwinTransformerBlock(embed_size=96, hidden_size=4*96, attn_heads=3, img_size=8, window_size=4)
    attn_mask, img_zones = swin.generate_shifted_attn_masks()
    if visualize:
        visualize_patterns(attn_mask, img_zones, title='(img_size=8x8, window_size=4x4)')

    expected_zones = torch.tensor([
        [ 0,  0,  0,  0,  1,  1,  5,  5],
        [ 0,  0,  0,  0,  1,  1,  5,  5],
        [ 0,  0,  0,  0,  1,  1,  5,  5],
        [ 0,  0,  0,  0,  1,  1,  5,  5],
        [ 2,  2,  2,  2,  3,  3,  7,  7],
        [ 2,  2,  2,  2,  3,  3,  7,  7],
        [10, 10, 10, 10, 11, 11, 15, 15],
        [10, 10, 10, 10, 11, 11, 15, 15]
    ])
    expected_masks = torch.tensor([
       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]],

       [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],

       [[0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0]],
    ])
    assert torch.all(attn_mask.int() == expected_masks)
    assert torch.all(img_zones == expected_zones)


    swin = SwinTransformerBlock(embed_size=96, hidden_size=4 * 96, attn_heads=96 // 32, img_size=6, window_size=3)
    attn_mask, img_zones = swin.generate_shifted_attn_masks()
    if visualize:
        visualize_patterns(attn_mask, img_zones, title='(img_size=6x6, window_size=3x3)')

    expected_zones = torch.tensor([
        [ 0,  0,  0,  1,  1,  5],
        [ 0,  0,  0,  1,  1,  5],
        [ 0,  0,  0,  1,  1,  5],
        [ 2,  2,  2,  3,  3,  7],
        [ 2,  2,  2,  3,  3,  7],
        [10, 10, 10, 11, 11, 15]
    ])
    expected_masks = torch.tensor([
        [[0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]],

        [[0, 0, 1, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 0, 1, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 1, 1, 0],
         [0, 0, 1, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 0, 1, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 1, 1, 0],
         [0, 0, 1, 0, 0, 1, 0, 0, 1],
         [0, 0, 1, 0, 0, 1, 0, 0, 1],
         [1, 1, 0, 1, 1, 0, 1, 1, 0]],

        [[0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 0, 0, 0]],

        [[0, 0, 1, 0, 0, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 1, 1, 1, 1],
         [1, 1, 0, 1, 1, 0, 1, 1, 1],
         [0, 0, 1, 0, 0, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 1, 1, 1, 1],
         [1, 1, 0, 1, 1, 0, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 0, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 0]]
    ])
    assert torch.all(attn_mask.int() == expected_masks)
    assert torch.all(img_zones == expected_zones)


def visualize_patterns(attn_mask, img_zones, title=''):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.matshow(img_zones)
    ax1.set_title('Windows pattern')

    grid = make_grid(attn_mask.unsqueeze(1), padding=2, pad_value=1, nrow=2).permute(1, 2, 0).float()
    ax2.matshow(grid.numpy())
    ax2.set_title('Attn patterns')
    ax2.axis(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

