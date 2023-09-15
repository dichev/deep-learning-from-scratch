import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid
from tqdm import trange
from datetime import datetime

from functions import init
from functions.activations import relu
from functions.losses import cross_entropy
from models.layers import Module, Linear, BatchNorm
from models import optimizers
from models.regularizers import L2_regularizer, grad_clip_, grad_clip_norm_
from preprocessing.floats import normalizeMinMax
from preprocessing.integer import one_hot
from preprocessing.dataset import data_split

now = datetime.now().strftime('%b%d %H-%M-%S')
train_writer = SummaryWriter(f'runs/{now} - train', flush_secs=2)
val_writer = SummaryWriter(f'runs/{now} - val', flush_secs=2)

# model hyperparams
n_features = 784  # train.data.shape[1] * train.data.shape[2]
n_hidden   = 10   # tmp. must be 100
n_classes  = 10   # len(train.classes)

# training hyperparams & settings
EPOCHS = 200
BATCH_SIZE = 1024
LEARN_RATE = 0.1
WEIGHT_DECAY = 0.001
DEVICE = 'cuda'


# Get data
train = datasets.MNIST('./data/', download=True, train=True)
test  = datasets.MNIST('./data/', download=True, train=False)
train_writer.add_image('images', make_grid(train.data[:100].unsqueeze(1).float(), 10, normalize=True), 0)

# Split data
X_train, y_train, X_val, y_val = data_split(train.data.view(-1, n_features), train.targets, (0.85, 0.15))
X_test, y_test = test.data.view(-1, n_features), test.targets

# Normalize and encode data
(x_min, x_max), num_classes = (0, 255), 10  # always reuse the normalization factors based on the training set
X_train = normalizeMinMax(X_train, x_min, x_max)
X_val   = normalizeMinMax(X_val, x_min, x_max)
X_test  = normalizeMinMax(X_test, x_min, x_max)
y_train = one_hot(y_train, num_classes)
y_val   = one_hot(y_val, num_classes)
y_test  = one_hot(y_test, num_classes)

# Add test/val sets to the device
X_val   = X_val.to(DEVICE)
X_test  = X_test.to(DEVICE)
y_val   = y_val.to(DEVICE)
y_test  = y_test.to(DEVICE)


# Model
class Net(Module):
    def __init__(self, input_size, output_size, hidden_size):  # over-parameterized model for testing purposes
        self.l1 = Linear(input_size, hidden_size, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.bn1 = BatchNorm(hidden_size, device=DEVICE)
        self.l2 = Linear(hidden_size, hidden_size//2, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.bn2 = BatchNorm(hidden_size//2, device=DEVICE)
        self.l3 = Linear(hidden_size//2, hidden_size//4, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.bn3 = BatchNorm(hidden_size//4, device=DEVICE)
        self.l4 = Linear(hidden_size//4, output_size, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.input_size, self.output_size = input_size, output_size

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.bn1.forward(x)
        x = relu(x)
        x = self.l2.forward(x)
        x = self.bn2.forward(x)
        x = relu(x)
        x = self.l3.forward(x)
        x = self.bn3.forward(x)
        x = relu(x)
        x = self.l4.forward(x)
        # x = softmax(x)
        return x

    @torch.no_grad()
    def evaluate(self, y_hat, y):
        predicted, actual = y_hat.argmax(1), y.argmax(1)
        correct = (predicted == actual)
        return correct.float().mean().item()


net = Net(n_features, n_classes, n_hidden)
net.summary()
net.export('../deeper/data/model.json')
optimizer = optimizers.SGD(net.parameters(), lr=LEARN_RATE)
# optimizer = optimizers.SGD_Momentum(net.parameters(), lr=LEARN_RATE, momentum=0.9)
# optimizer = optimizers.AdaGrad(net.parameters(), lr=LEARN_RATE)
# lr_scheduler = optimizers.LR_Scheduler(optimizer, decay=0.99, min_lr=1e-5)
# lr_scheduler = optimizers.LR_StepScheduler(optimizer, step_size=10, decay=0.99, min_lr=1e-5)
# lr_scheduler = optimizers.LR_PlateauScheduler(optimizer, patience=5, decay=0.99, min_lr=1e-5, threshold=1e-2)


# Training loop
N = len(y_train.data)
print(f'Fit {N} training samples in model: {net}')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    accuracy, loss, grad_norm = 0, 0, 0
    indices = torch.randperm(N)
    for i in range(0, N, BATCH_SIZE):
        batch = indices[i:i+BATCH_SIZE]
        y = y_train[batch].to(DEVICE)
        y_hat_logit = net.forward(X_train[batch].to(DEVICE))

        cost = cross_entropy(y_hat_logit, y, logits=True) # + L2_regularizer(net.parameters(), WEIGHT_DECAY)
        cost.backward()
        grad_norm_batch = grad_clip_norm_(net.parameters(), 0.9)
        optimizer.step().zero_grad()

        loss += cost.item() * len(batch) / N
        accuracy += net.evaluate(y_hat_logit, y) * len(batch) / N
        grad_norm += grad_norm_batch * len(batch) / N

    # Metrics
    train_writer.add_scalar('whp/Learn rate', optimizer.lr, epoch)
    train_writer.add_scalar('t/Loss', loss, epoch)
    train_writer.add_scalar('t/Accuracy', accuracy, epoch)
    train_writer.add_scalar('Gradients Norm', grad_norm, epoch)

    if epoch == 1 or epoch % 10 == 0:
        for name, param in net.parameters():
            train_writer.add_histogram(name.replace('.', '/'), param, epoch)

        with torch.no_grad():
            y_hat_logit = net.forward(X_val)
            val_loss = cross_entropy(y_hat_logit, y_val, logits=True).item()  # + L2_norm(net.parameters(), WEIGHT_DECAY).item()
            val_accuracy = net.evaluate(y_hat_logit, y_val)
            val_writer.add_scalar('t/Loss', val_loss, epoch)
            val_writer.add_scalar('t/Accuracy', val_accuracy, epoch)

    pbar.set_postfix(cost=f"{loss:.4f}|{val_loss:.4f}", accuracy=f"{accuracy:.4f}|{val_accuracy:.4f}", lr=optimizer.lr)
    # lr_scheduler.step(cost)

    # deeper.step()


with torch.no_grad():
    y_hat_logit = net.forward(X_test)
    test_loss = cross_entropy(y_hat_logit, y_test, logits=True).item()  # + L2_norm(net.parameters(), WEIGHT_DECAY).item()
    test_accuracy = net.evaluate(y_hat_logit, y_test)
    print(f'[Report only]: test_accuracy={test_accuracy:.4f}, test_cost={test_loss:.4f}')  # never tune hyperparams on the test set!


net.export('../deeper/data/model-trained.json')
