import torch
import torch.cuda
from torchvision import datasets
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, random_split
from tqdm import trange

from preprocessing.floats import img_unnormalize
from lib.functions.losses import cross_entropy, accuracy
from lib import optimizers
from lib.functions.activations import softmax
from models.convolutional_networks import SimpleCNN, LeNet5, AlexNet, NetworkInNetwork, VGG16, GoogLeNet
from utils import plots

# hyperparams & settings
EPOCHS = 10
BATCH_SIZE = 16
LEARN_RATE = 0.001
DEVICE = 'cuda'
SEED_DATA = 1111  # always reuse the same data seed for reproducibility and to avoid validation data leakage into the training set


# Data loaders
transforms = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),                      # normalized to [0,1]
    T.Normalize((.5, .5, .5), (.5, .5, .5)),  # shift and scale in the typical range (-1, 1)
])
train_dataset = datasets.CIFAR10('./data/', download=True, train=True, transform=transforms)
test_dataset  = datasets.CIFAR10('./data/', download=True, train=False, transform=transforms)
train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-2000, 2000))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Models
models = {
    'SimpleCNN': (SimpleCNN(n_classes=10), lambda x: x),
    'LeNet-5':   (LeNet5(n_classes=10), T.Grayscale(num_output_channels=1)),
    'AlexNet':   (AlexNet(n_classes=10), T.Resize((227, 227), antialias=True)),  # well, yeah..
    'NetworkInNetwork':   (NetworkInNetwork(n_classes=10), T.Resize((227, 227), antialias=True)),
    'VGG-16':   (VGG16(n_classes=10), T.Resize((224, 224))),
    'GoogLeNet': (GoogLeNet(n_classes=10), T.Resize((224, 224))),
}


@torch.no_grad()
def evaluate(model, loader, transform):
    total_loss = total_acc = 0
    n = len(loader)
    for X, y in loader:
        X, y = transform(X).to(DEVICE), y.to(DEVICE)
        z = model.forward(X)
        cost = cross_entropy(z, y, logits=True)
        total_acc += accuracy(z.argmax(dim=1), y) / n
        total_loss += cost.item() / n

    return total_loss, total_acc


def train(model, loader, transform):
    total_loss = total_acc = 0
    n = len(loader)  # n batch iterations
    pbar = trange(len(loader)*BATCH_SIZE, desc=f'Epoch (batch_size={BATCH_SIZE})')
    for i, (X, y) in enumerate(loader):
        X, y = transform(X).to(DEVICE), y.to(DEVICE)
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


# Training loop
for model_name, (model, transform) in models.items():
    model.to(DEVICE)
    optimizer = optimizers.SGD(model.parameters(), lr=LEARN_RATE)
    print(model.summary())
    print(f'Fit {len(train_dataset)} training samples in model: {model}')

    # Training
    for epoch in range(EPOCHS):
        loss, acc, pbar = train(model, train_loader, transform)
        val_loss, val_acc = evaluate(model, val_loader, transform)
        pbar.desc = f'Epoch {epoch + 1}/{EPOCHS}';
        pbar.set_postfix_str(f'{loss=:.3f}, {acc=:.3f} | {val_loss=:.3f}, {val_acc=:.3f}');
        pbar.close()

    test_loss, test_acc = evaluate(model, test_loader, transform)
    print(f'[Report only]: Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}')
    print(f'[Report only]: Failed on {int((1-test_acc)*len(test_loader))} samples out of {len(test_loader)}')


    # Plot some predictions
    N = 6
    X_sample, y_sample = next(iter(test_loader))
    X_sample = transform(X_sample)[:N]
    with torch.no_grad():
        out = model.forward(X_sample.to(DEVICE)).cpu()
    probs = softmax(out)
    images = img_unnormalize(X_sample)
    plots.img_topk(images, probs, test_dataset.classes, k=5, title=model_name)

