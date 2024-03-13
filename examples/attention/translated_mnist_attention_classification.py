import torch
from tqdm import trange

from models.attention_networks import RecurrentAttention
from lib.functions.losses import cross_entropy, accuracy
from lib.optimizers import Adam
from data.mnist import MNIST
from torchvision.transforms import v2 as T
from preprocessing.transforms import random_canvas_expand


# model hyperparams
focus_size = 8
k_focus_patches = 3
glimpses = 6

# training hyperparams & settings
EPOCHS = 100
BATCH_SIZE = 2 * 1024
LEARN_RATE = 0.1
DEVICE = 'cuda'


# Data loaders
train_loader, val_loader, test_loader = MNIST(transforms=[
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),                    # normalized to [0,1]
    lambda x: random_canvas_expand(x, width=60, height=60),  # translated MNIST: 28x28 -> 60x60 at random position
], batch_size=BATCH_SIZE)
X_sample, _ = next(iter(test_loader))               # non-varying sample used for visualizations
loc_sample = torch.Tensor(5, 2).uniform_(-1, 1)


# Model
print('WARNING! REINFORCE learning is not implemented yet.')
model = RecurrentAttention(focus_size=focus_size, steps=glimpses, k_focus_patches=k_focus_patches).to(DEVICE)
model.summary()
model.visualize(X_sample[:5].to(DEVICE), loc=loc_sample.to(DEVICE))
optimizer = Adam(model.parameters(), lr=LEARN_RATE)

@torch.no_grad()
def evaluate(model, loader):
    total_loss = total_acc = 0
    n = len(loader)
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        z, _ = model.forward(X)
        cost = cross_entropy(z, y, logits=True)
        total_acc += accuracy(z.argmax(dim=1), y) / n
        total_loss += cost.item() / n

    return total_loss, total_acc


def train(model, loader):
    total_loss = total_acc = 0
    n = len(loader)  # n batch iterations
    pbar = trange(len(loader)*BATCH_SIZE, desc=f'Epoch (batch_size={BATCH_SIZE})')
    for i, (X, y) in enumerate(loader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        z, locs = model.forward(X)  # todo: implement REINFORCE learning to the "locs"
        cost = cross_entropy(z, y, logits=True)
        cost.backward()
        optimizer.step()

        loss, acc = cost.item(), accuracy(z.argmax(dim=1), y)
        total_loss += loss / n
        total_acc += acc / n

        pbar.update(BATCH_SIZE)
        pbar.set_postfix_str(f'loss={loss:.3f}, acc={acc:.3f}')

    return total_loss, total_acc, pbar


# Training
for epoch in range(EPOCHS):
    loss, acc, pbar = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    pbar.desc = f'Epoch {epoch+1}/{EPOCHS}'; pbar.set_postfix_str(f'{loss=:.3f}, {acc=:.3f} | {val_loss=:.3f}, {val_acc=:.3f}'); pbar.close()

test_loss, test_acc = evaluate(model, test_loader)
print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}')
