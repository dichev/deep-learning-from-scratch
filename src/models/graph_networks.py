import torch
from lib.layers import Module, ModuleList, Linear, BatchNorm1d, ReLU, Sequential, Dropout, BatchAddPool, GCN_cell, GraphSAGE_cell
from lib.functions.activations import relu
from utils.other import identity

class GCN(Module):  # Graph Convolutional Network
    """
    Paper: Semi-Supervised Classification with Graph Convolutional Networks
    https://arxiv.org/pdf/1609.02907.pdf
    """
    def __init__(self, in_channels, hidden_size, n_classes, k_iterations=1, device='cpu'):
        self.layers = ModuleList([
            GCN_cell(in_channels if i == 0 else hidden_size, hidden_size, device)
            for i in range(k_iterations)]
        )
        self.head = Linear(hidden_size, n_classes)

    def forward(self, x, A):
        for graph in self.layers:
            x = graph.forward(x, A)
            x = relu(x)

        x = self.head.forward(x)
        return x


class GraphSAGE(Module):
    """
    Paper: Inductive Representation Learning on Large Graphs
    https://arxiv.org/pdf/1706.02216.pdf
    """
    def __init__(self, in_channels, hidden_size, n_classes, k_iterations=1, aggregation='maxpool', device='cpu'):
        self.layers = ModuleList([
            GraphSAGE_cell(in_channels if i == 0 else hidden_size * 2, hidden_size, aggregation, device)
            for i in range(k_iterations)]
        )
        self.head = Linear(self.layers[-1].out_channels, n_classes)

    def forward(self, x, A):
        for graph in self.layers:
            x = graph.forward(x, A)
            x = relu(x)
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        x = self.head.forward(x)
        return x


class GIN(Module):  # Graph Isomorphism Network
    """
    Paper: How Powerful are Graph Neural Networks?
    https://arxiv.org/pdf/1810.00826v3.pdf
    """

    def __init__(self, in_channels, hidden_size, n_classes, k_iterations=5, eps=0., device='cpu'):
        self.layers = ModuleList([
            Sequential(  # that is the MLP
                Linear(in_channels if i == 0 else hidden_size, hidden_size, device=device),
                BatchNorm1d(hidden_size, device=device),
                ReLU(),
                Linear(hidden_size, hidden_size, device=device),
                BatchNorm1d(hidden_size, device=device),
                ReLU(),
            )
            for i in range(k_iterations)
        ])
        self.add_pool = BatchAddPool()  # shared across layers, no parameters

        self.head = Sequential(  # note in the paper they use a separate linear+dropbox for each graph layer output
            Linear(hidden_size * k_iterations, hidden_size * k_iterations, device=device),
            ReLU(),
            Dropout(.5),
            Linear(hidden_size * k_iterations, n_classes, device=device),
        )

        self.eps = eps
        self.n_classes = n_classes

    def forward(self, X, A, batch_index=None):
        n, c = X.shape
        assert A.shape == (n, n)
        I = identity(n, sparse=A.is_sparse, device=X.device)

        features = []
        for layer in self.layers:
            # Aggregation - simply add neighbors and self features (summation is considered injective in contrast to mean or max)
            message = (A + (1 + self.eps) * I) @ X

            # Transform
            X = layer.forward(message)                      # (n, h)

            # Collect features as sum pool
            X_sums = self.add_pool.forward(X, batch_index)  # (batch_size, h)
            features.append(X_sums)

        # Classifier using the summed embeddings from each layer
        features = torch.cat(features, dim=1)               # (batch_size, n_layers * h)
        z = self.head.forward(features)                     # (n_classes)
        return z

