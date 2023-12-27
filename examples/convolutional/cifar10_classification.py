import torch
import torch.cuda
from torchvision import datasets
from tqdm import trange

from preprocessing.dataset import data_split
from preprocessing.integer import one_hot
from preprocessing.floats import image_normalize
from lib.functions.losses import cross_entropy, evaluate_accuracy, evaluate_accuracy_per_class
from lib.training import batches
from lib import optimizers
from models.convolutional_networks import SimpleCNN


# hyperparams & settings
img_shape = (32, 32, 3)  # train.data.shape
n_classes  = 10          # len(train.classes)
EPOCHS = 5
BATCH_SIZE = 64
LEARN_RATE = 0.1
DEVICE = 'cuda'
SEED_DATA = 1111  # always reuse the same data seed for reproducibility and to avoid validation data leakage into the training set


# Get data
train = datasets.CIFAR10('./data/', download=True, train=True)
test  = datasets.CIFAR10('./data/', download=True, train=False)
classes = train.classes

# Split data
X_train, y_train, X_val, y_val = data_split(train.data, train.targets, (0.90, 0.10), seed=SEED_DATA)
X_test, y_test = torch.tensor(test.data), torch.tensor(test.targets)

# Normalize and encode data
X_train = image_normalize(X_train)     # (N, C, W, H)
X_val   = image_normalize(X_val)       # (N, C, W, H)
X_test  = image_normalize(X_test)      # (N, C, W, H)
y_train = one_hot(y_train, n_classes)
y_val   = one_hot(y_val, n_classes)
y_test  = one_hot(y_test, n_classes)

net = SimpleCNN(device=DEVICE)
optimizer = optimizers.SGD_Momentum(net.parameters(), lr=LEARN_RATE, momentum=0.9)
print(net.summary())


# Training loop
N = len(y_train)
print(f'Fit {N} training samples in model: {net}')
pbar = trange(EPOCHS * ( 1 + N//BATCH_SIZE))
for epoch in range(1, EPOCHS+1):
    pbar.set_description(f"Epoch {epoch}/{EPOCHS}")
    accuracy, loss = 0, 0

    for i, (X, y, batch_fraction) in enumerate(batches(X_train, y_train, BATCH_SIZE, shuffle=True, device=DEVICE)):
        optimizer.zero_grad()

        y_hat_logit = net.forward(X)
        cost = cross_entropy(y_hat_logit, y, logits=True)
        cost.backward()
        optimizer.step()

        loss += cost.item()
        accuracy += evaluate_accuracy(y_hat_logit, y)
        pbar.update(1)
        if i % 100 == 99:
            pbar.set_postfix(cost=f"{loss/(i+1):.4f}", accuracy=f"{accuracy/(i+1):.4f}", lr=optimizer.lr)

    with torch.no_grad():
        y_hat_logit, targets = net.forward(X_val.to(DEVICE)), y_val.to(DEVICE)
        val_loss = cross_entropy(y_hat_logit, targets, logits=True).item()
        val_accuracy = evaluate_accuracy(y_hat_logit, targets)

    pbar.set_postfix(cost=f"{loss/(i+1):.4f}|{val_loss:.4f}", accuracy=f"{accuracy/(i+1):.4f}|{val_accuracy:.4f}", lr=optimizer.lr)
    pbar.write(f" Completed Epoch {epoch}")


with torch.no_grad():
    torch.cuda.empty_cache()
    y_hat_logit, targets = net.forward(X_test.to(DEVICE)), y_test.to(DEVICE)
    test_loss = cross_entropy(y_hat_logit, targets, logits=True).item()
    test_accuracy, test_accuracy_per_class = evaluate_accuracy_per_class(y_hat_logit, targets, classes)
    print(f'[Report only]: test_accuracy={test_accuracy:.4f}, test_cost={test_loss:.4f}')  # never tune hyperparams on the test set!
    print(f'[Report only]: Failed on {int((1-test_accuracy)*len(y_test))} samples out of {len(y_test)}')
    print(f'[Report only]: Accuracy per class:')
    for label in classes:
        print(f'  {test_accuracy_per_class[label] * 100:.1f}% {label:10s}')
    torch.cuda.empty_cache()
    print(f'{test_accuracy * 100:.1f}% Overall test accuracy')

