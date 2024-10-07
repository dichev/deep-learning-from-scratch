import torch
from tqdm import trange

from models.attention_networks import SpatialTransformerNet
from lib.functions.losses import cross_entropy
from lib.functions.metrics import accuracy
from lib.optimizers import SGD_Momentum
from torchvision.transforms import v2 as T
from data.mnist import MNIST


# training hyperparams & settings
EPOCHS = 20
BATCH_SIZE = 1024 * 2
LEARN_RATE = 0.01
DEVICE = 'cuda'


# Data loaders
train_loader, val_loader, test_loader = MNIST(transforms=[
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),                    # normalized to [0,1]
    T.RandomAffine(degrees=(-60, 60), translate=(0.3, 0.3), scale=(0.4, 1.0), interpolation=T.InterpolationMode.BILINEAR),
], batch_size=BATCH_SIZE)
X_sample, y_sample = next(iter(test_loader))  # non-varying sample used for visualizations

# Model
model = SpatialTransformerNet(n_classes=10, transformation_mode='affine').to(DEVICE)
model.summary()
optimizer = SGD_Momentum(model.parameters(), lr=LEARN_RATE)


@torch.no_grad()
def evaluate(model, loader):
    total_loss = total_acc = 0
    n = len(loader)
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        z = model.forward(X)
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
        z = model.forward(X)
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
model.spatial_transform.visualize(X_sample.to(DEVICE), y_sample.to(DEVICE), title='Start')
for epoch in range(EPOCHS):
    loss, acc, pbar = train(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    pbar.desc = f'Epoch {epoch+1}/{EPOCHS}'; pbar.set_postfix_str(f'{loss=:.3f}, {acc=:.3f} | {val_loss=:.3f}, {val_acc=:.3f}'); pbar.close()
    model.spatial_transform.visualize(X_sample.to(DEVICE), y_sample.to(DEVICE), title=f'Epoch {epoch+1}/{EPOCHS}')

test_loss, test_acc = evaluate(model, test_loader)
print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}')


