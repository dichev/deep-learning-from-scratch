import torch
import torch.cuda
from torchvision import datasets, transforms
from tqdm import trange

from preprocessing.dataset import data_split
from preprocessing.integer import one_hot
from preprocessing.floats import img_normalize, img_unnormalize
from lib.functions.losses import cross_entropy, evaluate_accuracy, evaluate_accuracy_per_class
from lib.training import batches
from lib import optimizers
from lib.functions.activations import softmax
from models.convolutional_networks import SimpleCNN, LeNet5, AlexNet
from utils import plots

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
X_train, y_train, X_val, y_val = data_split(train.data, train.targets, (len(train.data)-2000, 2000), seed=SEED_DATA)
X_test, y_test = torch.tensor(test.data), torch.tensor(test.targets)
# Normalize and encode data
X_train, X_val, X_test = [img_normalize(X) for X in (X_train, X_val, X_test)]  # (N, C, W, H)
y_train, y_val, y_test = [one_hot(y, n_classes) for y in (y_train, y_val, y_test)]   # (N, C)


models = {
    'SimpleCNN': (SimpleCNN(device=DEVICE), lambda x: x),
    'LeNet-5':   (LeNet5(device=DEVICE), transforms.Grayscale(num_output_channels=1)),
    # 'AlexNet':   (AlexNet(n_classes=10, device=DEVICE), transforms.Resize((227, 227), antialias=True))  # well, yeah..
}

for model_name, (net, adapt) in models.items():
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

            y_hat_logit = net.forward(adapt(X))
            cost = cross_entropy(y_hat_logit, y, logits=True)
            cost.backward()
            optimizer.step()

            loss += cost.item()
            accuracy += evaluate_accuracy(y_hat_logit, y)
            pbar.update(1)
            if i % 100 == 99:
                pbar.set_postfix(cost=f"{loss/(i+1):.4f}", accuracy=f"{accuracy/(i+1):.4f}", lr=optimizer.lr)

        with torch.no_grad():
            y_hat_logit, targets = net.forward(adapt(X_val.to(DEVICE))), y_val.to(DEVICE)
            val_loss = cross_entropy(y_hat_logit, targets, logits=True).item()
            val_accuracy = evaluate_accuracy(y_hat_logit, targets)

        pbar.set_postfix(cost=f"{loss/(i+1):.4f}|{val_loss:.4f}", accuracy=f"{accuracy/(i+1):.4f}|{val_accuracy:.4f}", lr=optimizer.lr)
        pbar.write(f" Completed Epoch {epoch}")

    with torch.no_grad():
        torch.cuda.empty_cache()
        y_hat_logit, targets = net.forward(adapt(X_test.to(DEVICE))), y_test.to(DEVICE)
        test_loss = cross_entropy(y_hat_logit, targets, logits=True).item()
        test_accuracy, test_accuracy_per_class = evaluate_accuracy_per_class(y_hat_logit, targets, classes)
        print(f'[Report only]: test_accuracy={test_accuracy:.4f}, test_cost={test_loss:.4f}')  # never tune hyperparams on the test set!
        print(f'[Report only]: Failed on {int((1-test_accuracy)*len(y_test))} samples out of {len(y_test)}')
        print(f'[Report only]: Accuracy per class:')
        for label in classes:
            print(f'  {test_accuracy_per_class[label] * 100:.1f}% {label:10s}')
        torch.cuda.empty_cache()
        print(f'{test_accuracy * 100:.1f}% Overall test accuracy')

    # Plot some predictions
    N = 6
    with torch.no_grad():
        out = net.forward(adapt(X_test[:N].to(DEVICE))).detach().cpu()
    probs = softmax(out)
    images = img_unnormalize(adapt(X_test[:N]))
    plots.img_topk(images, probs, classes, k=5, title=model_name)
