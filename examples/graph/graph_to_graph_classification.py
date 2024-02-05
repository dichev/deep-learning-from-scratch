import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from utils import plots

from models.graph_networks import GIN
from lib.functions.losses import cross_entropy, accuracy
from lib.optimizers import Adam
from utils.graph import edge_index_to_adj_matrix as to_adj_matrix


# hyperparams
GRAPH_LAYERS = 3
HIDDEN_CHANNELS = 64
LEARN_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 64
DEVICE = 'cuda'


# Data
dataset = TUDataset(root='./data', name='proteins').shuffle()
print(f'Dataset: {dataset}')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {dataset[0].x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

N = len(dataset)
train_loader = DataLoader(dataset[:int(N*.8)], batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(dataset[int(N*.8):int(N*.9)], batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset[int(N*.9):], batch_size=BATCH_SIZE, shuffle=False)


# Model
model = GIN(in_channels=dataset.num_features, hidden_size=HIDDEN_CHANNELS, n_classes=dataset.num_classes, n_layers=GRAPH_LAYERS, eps=0., device=DEVICE)
model.summary()
optimizer = Adam(model.parameters(), lr=LEARN_RATE)


@torch.no_grad()
def evaluate(model, loader):
    loss = acc = 0
    n = len(loader)
    for data in loader:
        data = data.to(DEVICE)
        X, A, y, batch_index = data.x, to_adj_matrix(data.edge_index), data.y, data.batch

        z = model.forward(X, A, batch_index)
        loss += cross_entropy(z, y, logits=True) / n
        acc += accuracy(z.argmax(dim=1), y) / n

    return loss, acc


for epoch in range(1, EPOCHS+1):
    loss = acc = val_loss = val_acc = 0
    n = len(train_loader)  # n batch iterations

    for data in train_loader:
        data = data.to(DEVICE)
        X, A, y, batch_index = data.x, to_adj_matrix(data.edge_index), data.y, data.batch

        optimizer.zero_grad()
        z = model.forward(X, A, batch_index)
        cost = cross_entropy(z, y, logits=True)
        loss += cost / n
        acc += accuracy(z.argmax(dim=1), y) / n
        cost.backward()
        optimizer.step()

    # Validation
    val_loss, val_acc = evaluate(model, val_loader)

    # Log
    print(f'Epoch {epoch:>3} | Train Loss: {loss:.2f} | Train Acc: {acc * 100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}%')

test_loss, test_acc = evaluate(model, test_loader)
print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')


# Plot some graphs from the test set (green correct / red incorrect)
with torch.no_grad():
    for data in test_loader:
        data = data.to(DEVICE)
        X, A, y, batch_index = data.x, to_adj_matrix(data.edge_index), data.y, data.batch
        z = model.forward(X, A, batch_index)
        correct_mask = z.argmax(dim=1) == y
        plots.graphs_grid(data[:24], correct_mask)
        break
