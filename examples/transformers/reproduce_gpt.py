import torch
from tqdm import trange
import tiktoken
from time import time
from data.edu_fineweb import DataLoaderLite, FineWebEduTokenizedDataset

from models.transformer_networks import GPT2
from lib.functions.losses import cross_entropy
from lib.optimizers import AdamW
from lib.regularizers import grad_clip_norm_
from utils.rng import seed_global
seed_global(1)


# Hyperparams
embed_size = 768
context_size = 1024
n_layers = 12
attn_heads = 12
vocab_size = 50_304  # 50_257

# Training config
batch_size = 8
epochs = 100
learn_rate = 3e-4
# todo: warmup
weight_decay = 0.1
device = 'cuda'
torch.set_float32_matmul_precision('high')  # use TFloat32 for multiplications outside the mixed-precision region


# Data
data = FineWebEduTokenizedDataset(data_root='./data/edu_fineweb10B')
train_loader = DataLoaderLite(data.train, batch_size, context_size)
val_loader = DataLoaderLite(data.val, batch_size, context_size)
tokenizer = tiktoken.get_encoding("gpt2")


# Models
model = GPT2(vocab_size, context_size, embed_size, hidden_size=4*embed_size, n_layers=n_layers, attn_heads=attn_heads)
model.to(device)
model.flash_attention(True)
optimizer = AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay, eps=1e-8, momentum=0.9, decay=0.95, weight_decay_filter=r'weight')
# optimizer = torch.optim.AdamW(model.parameters(named=False), lr=learn_rate, weight_decay=weight_decay, fused=True)
# model = torch.compile(model)


@torch.no_grad()
def generate_random_texts(model, prompt: str, attempts=5, max_tokens=50, use_cache=True):
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)]).expand(attempts, -1).to(device)
    outputs = model.generate(prompt_tokens, max_tokens, use_cache)
    for tokens in outputs:
        print(f'[{prompt}]' + tokenizer.decode(tokens.tolist()))

@torch.no_grad()
def evaluate(model, steps= 20):
    losses = 0
    val_loader.reset()
    for i in range(steps):
        context, targets = val_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # mixed precision to bfloat16
            logits = model(context.to(device))
            loss = cross_entropy(logits, targets.to(device), logits=True)
        losses += loss / steps
    return losses


# Training loop
print(f'Training {model.__class__.__name__}')
steps = len(train_loader)
for step in (pbar := trange(steps)):
    start_time = time()

    # Training step
    context, targets = train_loader.next_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):  # mixed precision to bfloat16
        logits = model(context.to(device))
        loss = cross_entropy(logits, targets.to(device), logits=True)
    loss.backward()
    grad_norm = grad_clip_norm_(model.parameters(), 1.)
    optimizer.step()
    optimizer.zero_grad()

    # Logs and evaluation
    pbar.update()
    pbar.set_postfix_str(f'tok/sek={(batch_size * context_size) / (time() - start_time):.1f} | {grad_norm=:.2f} | loss={loss:.4f} ')
    if step % 100 == 0:
        val_loss = evaluate(model)
        print(f'\nStep {step+1}/{steps} {val_loss=:.4f}')
        model.flash_attention(False)
        generate_random_texts(model, prompt='Hello I am not AGI yet, but ')
        model.visualize_attn_weights(subtitle=f'{step=}')
        model.flash_attention(True)

