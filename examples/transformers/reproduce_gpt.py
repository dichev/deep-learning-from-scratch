import torch
from tqdm import trange
import tiktoken
import time
from data.edu_fineweb import DataLoaderLite, FineWebEduTokenizedDataset

from models.transformer_networks import GPT2
from lib.functions.losses import cross_entropy
from lib.optimizers import AdamW, LR_CosineDecayScheduler
from lib.regularizers import grad_clip_norm_
from utils.other import format_seconds
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
batch_accum_steps = 524288 // (batch_size * context_size)  # to approximate the GPT2's much larger batch size (~0.5M tokens)
epochs = 100
warmup_steps = 5
learn_rate = 6e-3
learn_rate_min = 1e-5
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
optimizer = AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay, eps=1e-8, momentum=0.9, decay=0.95, weight_decay_filter=r'weight')
lr_scheduler = LR_CosineDecayScheduler(optimizer, warmup_steps=warmup_steps, decay_steps=epochs - warmup_steps, min_lr=learn_rate_min)
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
            logits = model(context.to(device), flash=True)
            loss = cross_entropy(logits, targets.to(device), logits=True)
        losses += loss / steps
    return losses


# Training loop
print(f'Training {model.__class__.__name__}')
steps = len(train_loader) // batch_accum_steps
for step in range(1, steps):
    start_time = time.time()

    # Training step
    loss_cum = 0
    for i in (pbar := trange(batch_accum_steps, desc='Grad accumulate')):
        st = time.time()
        context, targets = train_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # mixed precision to bfloat16
            logits = model(context.to(device), flash=True)
            loss = cross_entropy(logits, targets.to(device), logits=True) / batch_accum_steps
        loss.backward()
        loss_cum += loss.item()
        pbar.set_postfix_str(f'lr={optimizer.lr:.5e} | loss={loss*batch_accum_steps:.4f} | tok/sek={batch_size * context_size / (time.time() - st) :.1f} w/o update')

    grad_norm = grad_clip_norm_(model.parameters(), 1.)
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    # Logs and evaluation
    duration =  time.time() - start_time
    print(f'Step {step}/{steps}: lr={optimizer.lr:.5e} | {grad_norm=:.2f} | loss={loss_cum:.4f} | tok/sek={batch_size * context_size * batch_accum_steps / duration:.1f} | duration={duration:.1f} sec | ETA: {format_seconds(duration * (steps-step))}')
    if step == 1 or step % 100 == 0 or step == steps:
        val_loss = evaluate(model)
        print(f'\nStep {step}/{steps} {val_loss=:.4f}')
        generate_random_texts(model, prompt='Hello I am not AGI yet, but ')
        model.visualize_attn_weights(subtitle=f'{step=}')

