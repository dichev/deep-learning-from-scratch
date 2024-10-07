import torch
from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt

from lib.layers import Module, Linear, Graph_cell
from models.graph_networks import GCN, GraphSAGE
from lib.functions.activations import relu
from lib.functions.losses import cross_entropy
from lib.functions.metrics import accuracy
from lib.optimizers import Adam
from utils.graph import edge_index_to_adj_matrix as to_adj_matrix


# hyperparams
HIDDEN_CHANNELS = 6
LEARN_RATE = 0.01
EPOCHS = 100

# Graph dataset
dataset = KarateClub()
n = len(dataset[0].x)                    # total nodes
num_features = dataset.num_features + 6  # the one-hot features are equal to the num of nodes. Extending their dimensions for dimensionality sanity checks during debugging

X = torch.zeros(1, n, num_features)      # batch, nodes, features
X[:, :, :n] = dataset.x
y = dataset.y.view(1, n)
A = to_adj_matrix(dataset.edge_index, sparse=False).view(1, n, n)


# Define models
class GraphNet(Module):  # Baseline
    def __init__(self, input_channels, hidden_channels, n_classes):
        self.graph1 = Graph_cell(input_channels, hidden_channels)
        self.graph2 = Graph_cell(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, n_classes)

    def forward(self, x, A):
        x = relu(self.graph1.forward(x, A))
        x = relu(self.graph2.forward(x, A))
        x = self.out.forward(x)
        return x

models = {
    'GraphNet':  GraphNet(num_features, HIDDEN_CHANNELS, dataset.num_classes),
    'GraphConv': GCN(num_features, HIDDEN_CHANNELS, dataset.num_classes, k_iterations=2),
    'GraphSAGE': GraphSAGE(num_features, HIDDEN_CHANNELS, dataset.num_classes, k_iterations=2, aggregation='maxpool'),
}


# Overfit the different types of graph layers
for name, model in models.items():
    model.summary()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)

    for epoch in range(1, EPOCHS+1):
        optimizer.zero_grad()

        z = model.forward(X, A)
        loss = cross_entropy(z, y, logits=True)
        acc = accuracy(z.argmax(dim=-1), y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

    G = nx.DiGraph(A[0].to_dense().numpy())
    plt.figure(figsize=(7, 7))
    plt.title(f'{name} ({acc*100:.2f}% accuracy)')
    plt.axis(False)
    nx.draw_networkx(G, arrows=False, node_color=z[0].argmax(dim=-1))
    plt.show()

