import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid
from tqdm import trange
from datetime import datetime

from lib.functions import init
from lib.functions.activations import relu
from lib.functions.losses import cross_entropy, accuracy as evaluate_accuracy
from lib.layers import Module, Linear, Dropout
from lib import optimizers
from lib.regularizers import grad_clip_norm_, max_norm_constraint_
from lib.training import batches
from preprocessing.floats import normalizeMinMax
from preprocessing.integer import one_hot
from preprocessing.dataset import data_split

log_id = 'Dropout with max norm - '
now = datetime.now().strftime('%b%d %H-%M-%S')
train_writer = SummaryWriter(f'runs/{log_id}{now} - train', flush_secs=2)
val_writer = SummaryWriter(f'runs/{log_id}{now} - val', flush_secs=2)

# model hyperparams
n_features = 784  # train.data.shape[1] * train.data.shape[2]
n_classes  = 10   # len(train.classes)

# training hyperparams & settings
EPOCHS = 200
BATCH_SIZE = 1024
LEARN_RATE = 0.3
WEIGHT_DECAY = 0.001
DEVICE = 'cuda'
SEED_DATA = 1111  # always reuse the same data seed for reproducibility and to avoid validation data leakage into the training set

# Get data
train = datasets.MNIST('./data/', download=True, train=True)
test  = datasets.MNIST('./data/', download=True, train=False)
train_writer.add_image('images', make_grid(train.data[:100].unsqueeze(1).float(), 10, normalize=True), 0)

# Split data
X_train, y_train, X_val, y_val = data_split(train.data.view(-1, n_features), train.targets, (0.85, 0.15), seed=SEED_DATA)
X_test, y_test = test.data.view(-1, n_features), test.targets

# Normalize and encode data
(x_min, x_max), num_classes = (0, 255), 10  # always reuse the normalization factors based on the training set
X_train = normalizeMinMax(X_train, x_min, x_max)
X_val   = normalizeMinMax(X_val, x_min, x_max)
X_test  = normalizeMinMax(X_test, x_min, x_max)


# Model
class Net(Module):
    def __init__(self, input_size, output_size):  # over-parameterized model for testing purposes
        self.drop0 = Dropout(0.2)
        self.l1 = Linear(input_size, 300, device=DEVICE)
        self.drop1 = Dropout(0.5)
        # self.bn1 = BatchNorm(300, device=DEVICE)
        self.l2 = Linear(300, 200, device=DEVICE)
        self.drop2 = Dropout(0.5)
        # self.bn2 = BatchNorm(200, device=DEVICE)
        self.l3 = Linear(200, output_size, device=DEVICE)
        self.input_size, self.output_size = input_size, output_size

    def forward(self, x):
        x = self.drop0.forward(x)
        x = self.l1.forward(x)
        x = self.drop1.forward(x)
        # x = self.bn1.forward(x)
        x = relu(x)
        x = self.l2.forward(x)
        x = self.drop2.forward(x)
        # x = self.bn2.forward(x)
        x = relu(x)
        x = self.l3.forward(x)
        # x = softmax(x)
        return x


net = Net(n_features, n_classes)
net.summary()
# net.export('../deeper/data/model.json')
optimizer = optimizers.SGD(net.parameters(), lr=LEARN_RATE)
# optimizer = optimizers.SGD_Momentum(net.parameters(), lr=LEARN_RATE, momentum=0.9)
# optimizer = optimizers.AdaGrad(net.parameters(), lr=LEARN_RATE)
# lr_scheduler = optimizers.LR_Scheduler(optimizer, decay=0.99, min_lr=1e-5)
# lr_scheduler = optimizers.LR_StepScheduler(optimizer, step_size=10, decay=0.99, min_lr=1e-5)
# lr_scheduler = optimizers.LR_PlateauScheduler(optimizer, patience=5, decay=0.99, min_lr=1e-5, threshold=1e-2)


# Training loop
N = len(y_train)
print(f'Fit {N} training samples in model: {net}')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    accuracy, loss, grad_norm = 0, 0, 0

    for X, y, batch_fraction in batches(X_train, y_train, BATCH_SIZE, shuffle=True, device=DEVICE):
        optimizer.zero_grad()

        y_hat_logit = net.forward(X)
        cost = cross_entropy(y_hat_logit, y, logits=True)  # + elastic_regularizer(net.parameters(), WEIGHT_DECAY, 0.8)
        cost.backward()
        grad_norm_batch = grad_clip_norm_(net.parameters(), 0.9)
        optimizer.step()
        max_norm_constraint_(net.parameters(), 3.)

        loss += cost.item() * batch_fraction
        accuracy += evaluate_accuracy(y_hat_logit.argmax(-1), y) * batch_fraction
        grad_norm += grad_norm_batch * batch_fraction

    # Metrics
    train_writer.add_scalar('whp/Learn rate', optimizer.lr, epoch)
    train_writer.add_scalar('t/Loss', loss, epoch)
    train_writer.add_scalar('t/Accuracy', accuracy, epoch)
    train_writer.add_scalar('a/Gradients Norm', grad_norm, epoch)
    train_writer.add_scalar('a/Weights Norm', torch.tensor([p.norm(dim=0).mean() for name, p in net.parameters() if 'bias' not in name]).mean().item(), epoch)

    if epoch == 1 or epoch % 10 == 0:
        for name, param in net.parameters():
            train_writer.add_histogram(name.replace('.', '/'), param, epoch)

        with torch.no_grad():
            X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
            y_hat_logit = net.forward(X_val)
            val_loss = cross_entropy(y_hat_logit, y_val, logits=True).item() # + elastic_regularizer(net.parameters(), WEIGHT_DECAY, 0.8).item()
            val_accuracy = evaluate_accuracy(y_hat_logit.argmax(-1), y_val)
            val_writer.add_scalar('t/Loss', val_loss, epoch)
            val_writer.add_scalar('t/Accuracy', val_accuracy, epoch)

    pbar.set_postfix(cost=f"{loss:.4f}|{val_loss:.4f}", accuracy=f"{accuracy:.4f}|{val_accuracy:.4f}", lr=optimizer.lr)
    # lr_scheduler.step(cost)
    # deeper.step()


with torch.no_grad():
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
    y_hat_logit = net.forward(X_test)
    test_loss = cross_entropy(y_hat_logit, y_test, logits=True).item() # + elastic_regularizer(net.parameters(), WEIGHT_DECAY, 0.8).item()
    test_accuracy = evaluate_accuracy(y_hat_logit.argmax(-1), y_test)
    print(f'[Report only]: test_accuracy={test_accuracy:.4f}, test_cost={test_loss:.4f}')  # never tune hyperparams on the test set!
    print(f'[Report only]: Failed on {int((1-test_accuracy)*len(y_test))} samples out of {len(y_test)}')

# net.export('../deeper/data/model-trained.json')
