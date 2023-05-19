import torch
from matplotlib import pyplot as plt
from sklearn import datasets


def linearly_separable():
    X, y = datasets.make_blobs(
        n_samples=100, n_features=2, centers=2, cluster_std=1.15
    )

    def plot():
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.xlabel(f'$x_1$')
        plt.ylabel(f'$x_2$')

    return torch.Tensor(X), torch.Tensor(y), plot