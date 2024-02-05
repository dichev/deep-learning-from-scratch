import torch
from torch_geometric.datasets import KarateClub
import networkx as nx
import matplotlib.pyplot as plt

from lib.layers import GraphLayer, GraphConvLayer, GraphSAGELayer, Module, Linear
from lib.functions.activations import relu
from lib.functions.losses import cross_entropy
from lib.optimizers import Adam
from utils.graph import edge_index_to_adj_matrix as to_adj_matrix


# hyperparams
HIDDEN_CHANNELS = 4
LEARN_RATE = 0.1
EPOCHS = 100

# Graph dataset
dataset = KarateClub()
A = to_adj_matrix(dataset.edge_index)

# Define model factory
class GraphNet(Module):
    def __init__(self, GraphLayerClass):
        self.graph = GraphLayerClass(dataset.num_features, HIDDEN_CHANNELS)
        self.out = Linear(self.graph.out_channels, dataset.num_classes)

    def forward(self, x, A):
        x = self.graph.forward(x, A)
        x = relu(x)
        x = self.out.forward(x)
        return x


# Overfit the different types of graph layers
for GraphLayerType in (GraphLayer, GraphConvLayer, GraphSAGELayer):
    model = GraphNet(GraphLayerType)
    model.summary()
    optimizer = Adam(model.parameters(), lr=LEARN_RATE)

    for epoch in range(1, EPOCHS+1):
        optimizer.zero_grad()
        z = model.forward(dataset.x, A)
        loss = cross_entropy(z, dataset.y, logits=True)
        acc = (z.argmax(dim=1) == dataset.y).sum() / len(z)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

    G = nx.DiGraph(A.numpy())
    plt.figure(figsize=(7, 7))
    plt.title(f'{GraphLayerType.__name__} ({acc*100:.2f}% accuracy)')
    plt.axis(False)
    nx.draw_networkx(G, arrows=False, node_color=z.argmax(dim=1))
    plt.show()

