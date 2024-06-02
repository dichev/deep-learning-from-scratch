import pytest
import torch
from models.transformer_networks import LLaMA1
from utils.rng import seed_global


@pytest.mark.parametrize('batch_size', [1, 16])
@pytest.mark.parametrize('prompt_size', [1, 5, 100])
def test_LLAMA_caching(prompt_size, batch_size):
    vocab_size, context_size = 500, 200
    model = LLaMA1(vocab_size, context_size, embed_size=256, hidden_size=4*256, n_layers=4, attn_heads=4).to('cuda')
    prompt = torch.randint(vocab_size, (batch_size, prompt_size)).to('cuda')

    seed_global(1)
    tokens_cached = model.generate(prompt, max_tokens=context_size - prompt_size, use_cache=True)
    seed_global(1)
    tokens_not_cached = model.generate(prompt, max_tokens=context_size - prompt_size, use_cache=False)
    assert torch.allclose(tokens_cached, tokens_not_cached)