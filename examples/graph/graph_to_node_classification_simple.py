import torch
from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt

from lib.layers import Module, Linear, GraphLayer
from models.graph_networks import GCN, GraphSAGE
from lib.functions.activations import relu
from lib.functions.losses import cross_entropy, accuracy
from lib.optimizers import Adam
from utils.graph import edge_index_to_adj_matrix as to_adj_matrix


# hyperparams
HIDDEN_CHANNELS = 4
LEARN_RATE = 0.1
EPOCHS = 100

# Graph dataset
dataset = KarateClub()
A = to_adj_matrix(dataset.edge_index)

# Define models
class GraphNet(Module):  # Baseline
    def __init__(self, input_channels, hidden_channels, n_classes):
        self.graph1 = GraphLayer(input_channels, hidden_channels)
        self.graph2 = GraphLayer(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, n_classes)

    def forward(self, x, A):
        x = relu(self.graph1.forward(x, A))
        x = relu(self.graph2.forward(x, A))
        x = self.out.forward(x)
        return x

models = {
    'GraphNet':  GraphNet(dataset.num_features, HIDDEN_CHANNELS, dataset.num_classes),
    'GraphConv': GCN(dataset.num_features, HIDDEN_CHANNELS, dataset.num_classes, n_layers=2),
    'GraphSAGE': GraphSAGE(dataset.num_features, HIDDEN_CHANNELS, dataset.num_classes, n_layers=2, aggregation='maxpool'),
}


# Overfit the different types of graph layers
for name, model in models.items():
    model.summary()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)

    for epoch in range(1, EPOCHS+1):
        optimizer.zero_grad()
        z = model.forward(dataset.x, A)
        loss = cross_entropy(z, dataset.y, logits=True)
        acc = accuracy(z.argmax(dim=1), dataset.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

    G = nx.DiGraph(A.numpy())
    plt.figure(figsize=(7, 7))
    plt.title(f'{name} ({acc*100:.2f}% accuracy)')
    plt.axis(False)
    nx.draw_networkx(G, arrows=False, node_color=z.argmax(dim=1))
    plt.show()

