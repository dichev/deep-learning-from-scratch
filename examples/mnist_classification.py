import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid
from tqdm import trange
from datetime import datetime

from functions import init
from functions.activations import relu
from functions.losses import cross_entropy
from models.layers import Module, Linear
from models import optimizers
from models.regularizers import L2_norm
from preprocessing.floats import normalizeMinMax
from preprocessing.integer import one_hot

now = datetime.now().strftime('%b%d %H-%M-%S')
train_writer = SummaryWriter(f'runs/{now} - train', flush_secs=2)
test_writer = SummaryWriter(f'runs/{now} - test', flush_secs=2)

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


# Prepare date
train = datasets.MNIST('./data/', download=True, train=True)
test  = datasets.MNIST('./data/', download=True, train=False)  # ignored for now
train_writer.add_image('images', make_grid(train.data[:100].unsqueeze(1).float(), 10, normalize=True), 0)


X_train = normalizeMinMax(train.data.view(-1, n_features).float().to(DEVICE), dim=-1)
X_test  = normalizeMinMax(test.data.view(-1, n_features).float().to(DEVICE), dim=-1)
y_train = one_hot(train.targets).to(DEVICE)
y_test  = one_hot(test.targets).to(DEVICE)



# Model
class Net(Module):
    def __init__(self, input_size, output_size, hidden_size):  # over-parameterized model for testing purposes
        self.l1 = Linear(input_size, hidden_size, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.l2 = Linear(hidden_size, hidden_size//2, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.l3 = Linear(hidden_size//2, hidden_size//4, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.l4 = Linear(hidden_size//4, output_size, weights_init=init.kaiming_normal_relu, device=DEVICE)
        self.input_size, self.output_size = input_size, output_size

    def forward(self, x):
        x = self.l1.forward(x)
        x = relu(x)
        x = self.l2.forward(x)
        x = relu(x)
        x = self.l3.forward(x)
        x = relu(x)
        x = self.l4.forward(x)
        # x = softmax(x)
        return x

net = Net(n_features, n_classes, n_hidden)
net.summary()
net.export('../deeper/data/model.json')
optimizer = optimizers.SGD(net.parameters, lr=LEARN_RATE)
# lr_scheduler = optimizers.LR_Scheduler(optimizer, decay=0.99, min_lr=1e-5)
# lr_scheduler = optimizers.LR_StepScheduler(optimizer, step_size=10, decay=0.99, min_lr=1e-5)
lr_scheduler = optimizers.LR_PlateauScheduler(optimizer, patience=5, decay=0.99, min_lr=1e-5, threshold=1e-2)


# Training loop
N = len(train.data)
print(f'Fit {N} training samples in model: {net}')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    accuracy, loss = 0, 0
    indices = torch.randperm(N)
    for i in range(0, N, BATCH_SIZE):
        batch = indices[i:i+BATCH_SIZE]
        y = y_train[batch]
        y_hat_logit = net.forward(X_train[batch].view(-1, n_features).float())

        cost = cross_entropy(y_hat_logit, y, logits=True) + L2_norm(net.parameters(), WEIGHT_DECAY)
        cost.backward()
        optimizer.step().zero_grad()

        loss += cost.item() * len(batch) / N
        predicted, actual = y_hat_logit.argmax(1), y.argmax(1)
        accuracy += (predicted == actual).float().mean().item() * len(batch) / N

    # Metrics
    train_writer.add_scalar('whp/Learn rate', optimizer.lr, epoch)
    train_writer.add_scalar('t/Loss', loss, epoch)
    train_writer.add_scalar('t/Accuracy', accuracy, epoch)
    if epoch % 10 == 1:
        for name, param in net.parameters():
            train_writer.add_histogram(name.replace('.', '/'), param, epoch)

        with torch.no_grad():
            y_hat_logit = net.forward(X_test.view(-1, n_features).float())
            test_loss = cross_entropy(y_hat_logit, y_test, logits=True).item() + L2_norm(net.parameters(), WEIGHT_DECAY).item()
            test_accuracy = (y_hat_logit.argmax(1) == y_test.argmax(1)).float().mean().item()
            test_writer.add_scalar('t/Loss', test_loss, epoch)
            test_writer.add_scalar('t/Accuracy', test_accuracy, epoch)

    pbar.set_postfix(cost=f"{loss:.4f}|{test_loss:.4f}", accuracy=f"{accuracy:.4f}|{test_accuracy:.4f}", lr=optimizer.lr)
    lr_scheduler.step(cost)


net.export('../deeper/data/model-trained.json')
