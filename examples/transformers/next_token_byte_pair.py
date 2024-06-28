import torch
from torch.utils.data import DataLoader
from tqdm import trange
import math

from models.transformer_networks import GPT2, GPT3, LLaMA1, LLaMA2
from lib.functions.losses import cross_entropy
from lib.functions.metrics import accuracy
from lib.optimizers import AdamW, LR_CosineDecayScheduler
from lib.regularizers import grad_clip_norm_
from preprocessing.dataset import RandomTextDataset
from preprocessing.vocab import BPETokenizer
from utils.rng import seed_global
seed_global(1)


# Hyperparams
embed_size = 384
context_size = 256
n_layers = 6
attn_heads = 6
vocab_size = 300  # from which 256 tokens are reserved for character-level bytes

# Training config
batch_size = 64
epochs = 100
warmup_epochs = 5
learn_rate = 1e-3
learn_rate_min = 1e-5
weight_decay = 1e-2
device = 'cuda'
speedup = True
if speedup:
    torch.set_float32_matmul_precision('high')  # use TFloat32 for multiplications outside the mixed-precision region


# Models
models = {  # stored in RAM
    'GPT-2':     GPT2(vocab_size, context_size, embed_size, hidden_size=4*embed_size, n_layers=6, attn_heads=6, dropout=0),
    'GPT-3':     GPT3(vocab_size, context_size, embed_size, hidden_size=4*embed_size, n_layers=6, attn_heads=6, dropout=0, local_attn_block_size=8),
    'LLaMA-1': LLaMA1(vocab_size, context_size, embed_size, hidden_size=4*embed_size, n_layers=6, attn_heads=6),
    'LLaMA-2': LLaMA2(vocab_size, context_size, embed_size, hidden_size=4*embed_size, n_layers=6, attn_heads=6, attn_kv_groups=3),
}

# Prepare data
print('Data preprocessing..')
with open('./data/shakespeare.txt', 'r', encoding='utf8') as f:
    text = f.read()

print('Training BPE tokenizer:')
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size-256)
assert len(tokenizer.vocab) == vocab_size

print('Encode data..')
encoded = tokenizer.encode(text)
assert tokenizer.decode(encoded) == text
encoded = torch.tensor(encoded)

print('Split training data')
split = int(len(encoded) * 0.9)
train_data = RandomTextDataset(encoded[:split], context_size)
val_data = RandomTextDataset(encoded[split:], context_size)
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)



@torch.no_grad()
def generate_random_text(model, prompt='', max_tokens=100, use_cache=True):
    if prompt:
        print(f'[{prompt}]', end='')
        prompt = torch.tensor([tokenizer.encode(prompt)])
    else:
        prompt = torch.randint(vocab_size, (1, 1))
    tokens = model.generate(prompt.to(device), max_tokens, use_cache)
    print(tokenizer.decode(tokens.ravel().tolist()))


@torch.no_grad()
def evaluate(model, loader, max_iter=None):
    losses, acc = [], []
    for i, (context, targets) in enumerate(loader):
        context, targets = context.to(device), targets.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=speedup):  # mixed precision to bfloat16
            logits = model(context)
            loss = cross_entropy(logits, targets, logits=True)
        losses.append(loss)
        acc.append(accuracy(logits.argmax(dim=-1), targets))
        if max_iter and i >= max_iter: break
    return torch.stack(losses).mean(), torch.stack(acc).mean()


def train(model, optimizer, loader, desc=''):
    pbar = trange(len(loader), desc=desc)
    for context, targets in loader:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=speedup):  # mixed precision to bfloat16
            logits = model(context.to(device))
            loss = cross_entropy(logits, targets.to(device), logits=True)
        loss.backward()
        grad_clip_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()
        pbar.update()
        pbar.set_postfix_str(f'loss={loss:.4f} ({loss/math.log(2):.2f} bits per byte)')
    pbar.close()


# Training loop
for name, model in models.items():
    print(f'Training {name}')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay, weight_decay_filter=r'weight')
    lr_scheduler = LR_CosineDecayScheduler(optimizer, warmup_steps=warmup_epochs, decay_steps=epochs-warmup_epochs, min_lr=learn_rate_min)

    # Training
    for epoch in range(1, epochs+1):
        train(model, optimizer, train_loader, desc=f'Epoch {epoch}/{epochs}')
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch}/{epochs} | lr={optimizer.lr:.5e} | {train_loss=:.4f} {val_loss=:.4f} | {train_acc=:.4f} {val_acc=:.4f}')
        lr_scheduler.step()
        if epoch < 10 or epoch % 10 == 0:
            model.visualize_attn_weights(subtitle=f'{epoch=}')
            generate_random_text(model, max_tokens=100)

    generate_random_text(model, max_tokens=100, prompt='Now that we have learned how to work with probability')
