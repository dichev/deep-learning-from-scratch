import torch
from lib.layers import Module, ModuleList, Linear, BatchNorm, ReLU, Sequential, Dropout, BatchAddPool, GCN_cell, GraphSAGE_cell, DiffPool
from lib.functions.activations import relu
from utils.other import identity

class GCN(Module):  # Graph Convolutional Network
    """
    Paper: Semi-Supervised Classification with Graph Convolutional Networks
    https://arxiv.org/pdf/1609.02907.pdf
    """
    def __init__(self, in_channels, hidden_size, n_classes=None, k_iterations=1):
        self.layers = ModuleList([
            GCN_cell(in_channels if i == 0 else hidden_size, hidden_size)
            for i in range(k_iterations)]
        )
        self.project = n_classes is not None
        if self.project:
            self.head = Linear(hidden_size, n_classes)

    def forward(self, x, A):
        for graph in self.layers:
            x = graph.forward(x, A)
            x = relu(x)

        if self.project:
            x = self.head.forward(x)
        return x


class GraphSAGE(Module):
    """
    Paper: Inductive Representation Learning on Large Graphs
    https://arxiv.org/pdf/1706.02216.pdf
    """
    def __init__(self, in_channels, hidden_size, n_classes=None, k_iterations=1, aggregation='maxpool'):
        self.layers = ModuleList([
            GraphSAGE_cell(in_channels if i == 0 else hidden_size * 2, hidden_size, aggregation)
            for i in range(k_iterations)]
        )
        self.project = n_classes is not None
        if self.project:
            self.head = Linear(self.layers[-1].out_channels, n_classes)

    def forward(self, x, A):
        for graph in self.layers:
            x = graph.forward(x, A)
            x = relu(x)
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        if self.project:
            x = self.head.forward(x)
        return x


class GIN(Module):  # Graph Isomorphism Network
    """
    Paper: How Powerful are Graph Neural Networks?
    https://arxiv.org/pdf/1810.00826v3.pdf
    """

    def __init__(self, in_channels, hidden_size, n_classes, k_iterations=5, eps=0.):
        self.layers = ModuleList([
            Sequential(  # that is the MLP
                Linear(in_channels if i == 0 else hidden_size, hidden_size),
                BatchNorm(hidden_size, batch_dims=(0, 1)),
                ReLU(),
                Linear(hidden_size, hidden_size),
                BatchNorm(hidden_size, batch_dims=(0, 1)),
                ReLU(),
            )
            for i in range(k_iterations)
        ])
        self.add_pool = BatchAddPool()  # shared across layers, no parameters

        self.head = Sequential(  # note in the paper they use a separate linear+dropbox for each graph layer output
            Linear(hidden_size * k_iterations, hidden_size * k_iterations),
            ReLU(),
            Dropout(.5),
            Linear(hidden_size * k_iterations, n_classes),
        )

        self.eps = eps
        self.n_classes = n_classes

    def forward(self, X, A, batch_index=None):
        b, n, c = X.shape  # batch_size, nodes, channel(features)
        assert A.shape == (b, n, n)

        # cache a "soft-self" adjacency matrix
        I = identity(n, sparse=A.is_sparse, device=X.device)
        A_self = (A + (1 + self.eps) * I)

        features = []
        for layer in self.layers:
            # Aggregation - simply add neighbors and self features (summation is considered injective in contrast to mean or max)
            message = A_self @ X                      # (b, n, c) -> (b, n, c)

            # Transform
            X = layer.forward(message)                # (b, n, h)

            # Collect features as sum pool
            X_sums = X.sum(dim=1)                     # (b, h)
            features.append(X_sums)

        # Classifier using the summed embeddings from each layer
        features = torch.cat(features, dim=-1)        # (b, h * num_layers)
        z = self.head.forward(features)               # (n_classes)
        return z, (0, 0)  # ignore the zeros, there are for compatibility with DiffPoolNet output


class DiffPoolNet(Module):
    """
    Paper: Hierarchical Graph Representation Learning with Differentiable Pooling
    https://proceedings.neurips.cc/paper_files/paper/2018/file/e77dbaf6759253c7c6d0efc5690369c7-Paper.pdf
    """
    def __init__(self, in_channels, embed_size, n_clusters=(None, None), n_classes=1):
        self.n_clusters = n_clusters

        self.gcn1_embed = GraphSAGE(in_channels, embed_size // 2, k_iterations=2, aggregation='mean')  # note graphSage output is concatenated, thus the out_size will be embed_size
        self.gcn1_assign = GraphSAGE(in_channels, n_clusters[0], k_iterations=2, aggregation='mean')   # [paper] they used BatchNorm, but here simple normalization is performed across the channels
        self.pool1 = DiffPool()

        self.gcn2_embed = GraphSAGE(embed_size, embed_size // 2, k_iterations=3, aggregation='mean')
        self.gcn2_assign = GraphSAGE(embed_size, n_clusters[1], k_iterations=3, aggregation='mean')
        self.pool2 = DiffPool()

        self.gcn3_embed = GraphSAGE(embed_size, embed_size // 2, k_iterations=3, aggregation='mean')
        # self.pool_final = DiffPool()

        self.head = Sequential(
            Linear(embed_size, embed_size),
            ReLU(),
            Linear(embed_size, n_classes)
        )

    def forward(self, X, A):
        b, n, c = X.shape

        Z = self.gcn1_embed.forward(X, A)                         # node embeddings
        S = self.gcn1_assign.forward(X, A)                        # cluster assignment matrix (softmax is done by the pool)
        X, A, (loss_l1, loss_e1) = self.pool1.forward(Z, A, S)    # A is no more a sparse binary matrix, but a weighted dense matrix that represents the connectivity strength between each cluster

        Z = self.gcn2_embed.forward(X, A)                         # node embeddings
        S = self.gcn2_assign.forward(X, A)                        # cluster assignment matrix (softmax is done by the pool)
        X, A, (loss_l2, loss_e2) = self.pool2.forward(Z, A, S)

        Z = self.gcn3_embed.forward(X, A)                         # node embeddings
        X = Z.sum(dim=1)                                          # global add pooling - it is equivalent to the final pool (from paper), where S is a vector of 1's, thus all nodes are always assigned to a single cluster

        X = self.head.forward(X)

        # Collect the pooling losses
        loss_link, loss_entropy = loss_l1 + loss_l2, loss_e1 + loss_e2

        return X, (loss_link, loss_entropy)
