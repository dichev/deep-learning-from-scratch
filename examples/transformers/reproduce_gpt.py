import torch
from tqdm import trange
import tiktoken
import time
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from data.edu_fineweb import DataLoaderLite, FineWebEduTokenizedDataset

from models.transformer_networks import GPT2
from lib.functions.losses import cross_entropy
from lib.optimizers import AdamW, LR_CosineDecayScheduler
from lib.regularizers import grad_clip_norm_
from utils.other import format_seconds
from utils.rng import seed_global, set_rng_states, get_rng_states
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
learn_rate = 6e-4
learn_rate_min = 6e-5
warmup_steps = 715
weight_decay = 0.1
device = 'cuda'
torch.set_float32_matmul_precision('high')  # use TFloat32 for multiplications outside the mixed-precision region

# Reproducing
checkpoint_step = 0  # load check point from here
checkpoint_every = 1000  # steps
checkpoint_dir = './runs/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(f'runs/GPT_Reproduce - {datetime.now().strftime("%b%d %H-%M-%S")}', flush_secs=2)


# Data
data = FineWebEduTokenizedDataset(data_root='./data/edu_fineweb10B')
train_loader = DataLoaderLite(data.train, batch_size, context_size)
val_loader = DataLoaderLite(data.val, batch_size, context_size)
tokenizer = tiktoken.get_encoding("gpt2")
steps = len(train_loader) // batch_accum_steps
assert steps == 19073  # expected steps for 1 epoch

# Models
model = GPT2(vocab_size, context_size, embed_size, hidden_size=4*embed_size, n_layers=n_layers, attn_heads=attn_heads)
model.to(device)
optimizer = AdamW(model.parameters(), lr=learn_rate, weight_decay=weight_decay, eps=1e-8, momentum=0.9, decay=0.95, weight_decay_filter=r'weight')
lr_scheduler = LR_CosineDecayScheduler(optimizer, warmup_steps=warmup_steps, decay_steps=steps - warmup_steps, min_lr=learn_rate_min)
# optimizer = torch.optim.AdamW(model.parameters(named=False), lr=learn_rate, weight_decay=weight_decay, fused=True)
# model = torch.compile(model)


@torch.no_grad()
def generate_random_texts(model, prompt: str, attempts=5, max_tokens=50, use_cache=True):
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)]).expand(attempts, -1).to(device)
    outputs = model.generate(prompt_tokens, max_tokens, use_cache)
    for tokens in outputs:
        tokens = tokens[tokens<=tokenizer.max_token_value]  # because the vocab size was extended from 50_257 to 50_304
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

@torch.no_grad()
def save_checkpoint(step, loss):
    path = f'{checkpoint_dir}/reproduce-gpt_step_{step:05}.pt'
    print(f'Saving checkpoint at {step=}...')
    torch.save({
        'step': step,
        'loss': loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'seed': {  # to reproduce from the same seed state (very useful for debugging)
            'train_loader': train_loader.get_state(),
            'val_loader': val_loader.get_state(),
            'rng': get_rng_states(),
        }
    }, path)


@torch.no_grad()
def load_checkpoint(step):
    print(f'Loading checkpoints at {step=}..')
    state = torch.load(f'{checkpoint_dir}/reproduce-gpt_step_{step:05}.pt')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    lr_scheduler.load_state_dict(state['lr_scheduler'])
    train_loader.set_state(**state['seed']['train_loader'])
    val_loader.set_state(**state['seed']['val_loader'])
    set_rng_states(state['seed']['rng'])
    print(f'Loaded checkpoint from step {state["step"]} (loss = {state["loss"]:.4f})')


# Load checkpoint
if checkpoint_step > 0:
    load_checkpoint(checkpoint_step)
print(f'Initial val loss: {evaluate(model):.4f}')


# Training loop
print(f'Training...')
for step in range(1 + checkpoint_step, steps):
    start_time = time.time()

    # Training step
    loss_cum = 0
    for i in (pbar := trange(batch_accum_steps, desc=f'Step {step}/{steps} (Grad accumulate)', leave=False)):
        st = time.time()
        context, targets = train_loader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # mixed precision to bfloat16
            logits = model(context.to(device), flash=True)
            loss = cross_entropy(logits, targets.to(device), logits=True) / batch_accum_steps
        loss.backward()
        loss_cum += loss.item()
        pbar.set_postfix_str(f'loss={loss*batch_accum_steps:.4f} | tok/sek={batch_size * context_size / (time.time() - st) :.1f} w/o update')

    grad_norm = grad_clip_norm_(model.parameters(), 1.)
    optimizer.step()
    optimizer.zero_grad()
    lr_scheduler.step()

    # Logs and evaluation
    duration = time.time() - start_time
    print(f'\rStep {step}/{steps}: lr={optimizer.lr:.5e} | {grad_norm=:.4f} | loss={loss_cum:.6f} | tok/sek={batch_size * context_size * batch_accum_steps / duration:.1f} | duration={duration:.1f} sec | ETA: {format_seconds(duration * (steps-step))}')
    writer.add_scalar('t/Learn rate', optimizer.lr, step)
    writer.add_scalar('t/Train Loss', loss_cum, step)
    writer.add_scalar('a/Gradients Norm', grad_norm, step)
    writer.add_scalar('a/Weights Norm', model.weight_norm(), step)
    if step in [1, 2, 5, 10, 50, 100, steps] or step % checkpoint_every == 0:
        val_loss = evaluate(model)
        print(f'Step {step}/{steps}: {val_loss=:.4f}')
        writer.add_scalar('t/Val loss', val_loss, step)
        generate_random_texts(model, prompt='Hello I am not AGI yet, but ')
        model.visualize_attn_weights(subtitle=f'{step=}')

        save_checkpoint(step, loss_cum)
        # load_checkpoint(step=1) # for debugging


