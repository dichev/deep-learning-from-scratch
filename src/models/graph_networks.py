import torch
from lib.layers import Module, ModuleList, Linear, BatchNorm1d, ReLU, Sequential, Dropout, GraphAddLayer, BatchAddPool


class GIN(Module):  # Graph Isomorphism Network
    """
    Paper: How Powerful are Graph Neural Networks?
    https://arxiv.org/pdf/1810.00826v3.pdf
    """

    def __init__(self, in_channels, hidden_size, n_classes, n_layers=5, eps=0., device='cpu'):
        self.aggregate = GraphAddLayer()  # shared across layers, no parameters
        self.layers = ModuleList([
            Sequential(  # that is the MLP
                Linear(in_channels if i == 0 else hidden_size, hidden_size, device=device),
                BatchNorm1d(hidden_size, device=device),
                ReLU(),
                Linear(hidden_size, hidden_size, device=device),
                BatchNorm1d(hidden_size, device=device),
                ReLU(),
            )
            for i in range(n_layers)
        ])
        self.add_pool = BatchAddPool()  # shared across layers, no parameters

        self.head = Sequential(  # note in the paper they use a separate linear+dropbox for each graph layer output
            Linear(hidden_size * n_layers, hidden_size * n_layers, device=device),
            ReLU(),
            Dropout(.5),
            Linear(hidden_size * n_layers, n_classes, device=device),
        )

        self.eps = eps
        self.n_classes = n_classes

    def forward(self, X, A, batch_index=None):
        n, c = X.shape
        assert A.shape == (n, n)

        features = []
        for layer in self.layers:
            # Aggregate neighbors and self features
            message = self.aggregate.forward(X, A)          # (n, c)

            # Transform
            X = layer.forward(message)                      # (n, h)

            # Collect features as sum pool
            X_sums = self.add_pool.forward(X, batch_index)  # (batch_size, h)
            features.append(X_sums)

        # Classifier using the summed embeddings from each layer
        features = torch.cat(features, dim=1)               # (batch_size, n_layers * h)
        z = self.head.forward(features)                     # (n_classes)
        return z


