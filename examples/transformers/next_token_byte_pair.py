import torch
from torch.utils.data import DataLoader
from tqdm import trange

from models.transformer_networks import GPT2
from lib.functions.losses import cross_entropy
from lib.functions.metrics import accuracy
from lib.optimizers import AdamW
from preprocessing.dataset import RandomTextDataset
from preprocessing.vocab import BPETokenizer
from utils.rng import seed_global
seed_global(1)


# Hyperparams
context_size = 256
batch_size = 64
epochs = 100
learn_rate = 1e-3
device = 'cuda'


# Prepare data
print('Data preprocessing..')
with open('./data/shakespeare.txt', 'r', encoding='utf8') as f:
    text = f.read()

tokenizer = BPETokenizer()
print('Training BPE tokenizer:')
tokenizer.train(text, 200)
assert tokenizer.decode(tokenizer.encode(text)) == text
encoded = torch.tensor(tokenizer.encode(text))
split = int(len(encoded) * 0.9)
train_data = RandomTextDataset(encoded[:split], context_size)
val_data = RandomTextDataset(encoded[split:], context_size)
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)



# Model
model = GPT2(vocab_size=len(tokenizer.vocab), context_size=256, embed_size=384, hidden_size=4*384, n_layers=4, attn_heads=4, dropout=.2, padding_idx=None).to(device)
optimizer = AdamW(model.parameters(), lr=learn_rate)


@torch.no_grad()
def generate_random_text(model, prompt=None, max_tokens=100):
    prompt = prompt if prompt is not None else torch.tensor([[2]]).to(device)  # note: not using the padding token
    print(f"[{tokenizer.decode(prompt.flatten().detach().tolist())}]", end='')
    tokens = [idx.item() for idx in model.generate(prompt, max_tokens)]
    print(tokenizer.decode(tokens))


@torch.no_grad()
def evaluate(model, loader, max_iter=None):
    losses, acc = [], []
    for i, (context, targets) in enumerate(loader):
        context, targets = context.to(device), targets.to(device)
        logits = model(context)
        loss = cross_entropy(logits, targets, logits=True)
        losses.append(loss)
        acc.append(accuracy(logits.argmax(dim=-1), targets))
        if max_iter and i >= max_iter: break
    return torch.stack(losses).mean(), torch.stack(acc).mean()


def train(model, optimizer, loader):
    pbar = trange(len(loader), desc=f'Epoch {epoch}/{epochs}')
    for context, targets in loader:
        logits = model(context.to(device))
        loss = cross_entropy(logits, targets.to(device), logits=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.update()
        pbar.set_postfix_str(f'loss={loss:.4f}')
    pbar.close()


# Training
for epoch in range(1, epochs+1):
    train(model, optimizer, train_loader)
    train_loss, train_acc = evaluate(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f'Epoch {epoch}/{epochs} {train_loss=:.4f} {val_loss=:.4f} | {train_acc=:.4f} {val_acc=:.4f}')
    if epoch < 10 or epoch % 10 == 0:
        generate_random_text(model, max_tokens=100)
        model.visualize_attn_weights(subtitle=f'{epoch=}')
