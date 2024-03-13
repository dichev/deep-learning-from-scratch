import torch
from math import ceil
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
import torch_geometric.transforms as T

from utils import plots
from models.graph_networks import GIN, DiffPoolNet
from lib.functions.losses import cross_entropy, accuracy
from lib.optimizers import Adam
from utils.graph import edge_index_to_adj_list as to_adj_list


# hyperparams
HIDDEN_CHANNELS = 64
LEARN_RATE = 0.01
EPOCHS = 20
BATCH_SIZE = 32
DEVICE = 'cuda'


# Data
max_nodes = 200  # only 12 from 1113 graphs in the dataset has more nodes
dataset = TUDataset(
    root='./data/PROTEINS-dense', name='PROTEINS',
    transform=T.ToDense(max_nodes),                 # make each graph with the same number of nodes (to allow batching) by adding nodes with zero connections
    pre_filter=lambda d: d.num_nodes <= max_nodes   # remove the graphs with more than max_nodes
)
print(f'Dataset: {dataset}')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {dataset[0].x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

N = len(dataset)
train_loader = DenseDataLoader(dataset[:int(N*.8)], batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DenseDataLoader(dataset[int(N*.8):int(N*.9)], batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DenseDataLoader(dataset[int(N*.9):], batch_size=BATCH_SIZE, shuffle=False)


@torch.no_grad()
def evaluate(model, loader):
    acc = 0
    n = len(loader)
    for data in loader:
        data = data.to(DEVICE)
        z, _ = model.forward(data.x, data.adj)
        acc += accuracy(z.argmax(dim=1), data.y.view(-1)) / n

    return loss, acc

def train(model, loader):
    loss = acc = 0
    n = len(loader)  # n batch iterations
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        z, (loss_link, loss_entropy) = model.forward(data.x, data.adj)
        cost = cross_entropy(z, data.y.view(-1), logits=True) + loss_link + loss_entropy
        cost.backward()
        optimizer.step()

        loss += cost / n
        acc += accuracy(z.argmax(dim=1), data.y.view(-1)) / n

    return loss, acc


# Model
models = {
    'GIN': GIN(in_channels=dataset.num_features, hidden_size=HIDDEN_CHANNELS, n_classes=dataset.num_classes, k_iterations=3, eps=0.),
    'DiffPool': DiffPoolNet(in_channels=dataset.num_features, embed_size=HIDDEN_CHANNELS, n_clusters=(ceil(max_nodes*.25), ceil(max_nodes*.25*.25)), n_classes=dataset.num_classes)
}

# Training
for name, model in models.items():
    model.to(DEVICE)
    model.summary()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)
    for epoch in range(1, EPOCHS+1):
        loss, acc = train(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        print(f'Epoch {epoch:>3} | Train Total Loss: {loss:.2f} | Train Acc: {acc * 100:>5.2f}% | Val Acc: {val_acc * 100:.2f}%')

    test_loss, test_acc = evaluate(model, test_loader)
    print(f'Test Acc: {test_acc * 100:.2f}%')

    # Plot some graphs from the test set (green correct / red incorrect)
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            X, A, y = data.x, data.adj, data.y.view(-1)
            z, _ = model.forward(X, A)
            correct_mask = z.argmax(dim=1) == y
            graphs = [to_adj_list(adj.to_sparse().indices()) for adj in data.adj]
            plots.graphs_grid(graphs[:16], correct_mask[:16], title=name)
            break

