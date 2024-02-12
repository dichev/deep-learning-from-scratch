import torch
from matplotlib import pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import math

def decision_boundary_2d(X, Y, classifier):
    xx, yy = torch.meshgrid(torch.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                            torch.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500), indexing='xy')

    Z = classifier(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=-1))
    Z = Z.view(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap='Blues')
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap='Blues')
    plt.xlabel(f'$x_1$')
    plt.ylabel(f'$x_2$')
    plt.title('Decision boundary')


def decision_boundary_3d(X, Y, classifier):
    xx, yy = torch.meshgrid(torch.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                            torch.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100), indexing='xy')

    Z = classifier(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=-1))
    Z = Z.view(xx.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], Y, c=Y, edgecolors='k', cmap='coolwarm')  # Original data points
    ax.plot_surface(xx.detach(), yy.detach(), Z.detach(), alpha=0.5, cmap='coolwarm')  # Decision boundary surface

    ax.set_xlabel(f'$x_1$')
    ax.set_ylabel(f'$x_2$')
    ax.set_zlabel('Class')
    plt.title('Decision boundary')

def LaTeX(title, expression):
    plt.title(title)
    plt.text(0.1, 0.2, expression, size=20)
    plt.axis('off')
    plt.show()

def img_topk(images, probs, labels, k=5, title=''):
    N = len(images)
    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(N, 2, figsize=(3,N*1.3))
    plt.suptitle(title)
    for i in range(N):
        top5 = probs[i].topk(k).indices

        # Plot image
        axs[i, 0].imshow(images[i], cmap='gray')
        axs[i, 0].axis('off')

        # Plot top5 probs
        y_pos = range(k)
        axs[i, 1].barh(y_pos, probs[i][top5], align='center', color='green')
        axs[i, 1].set_yticks(y_pos)
        axs[i, 1].set_yticklabels([labels[idx] for idx in top5])
        axs[i, 1].invert_yaxis()
        axs[i, 1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


def graph_as_spring_2d(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, edge_color='gray')
    plt.show()


def graph_as_spring_3d(G):
    pos = nx.spring_layout(G, dim=3)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d")
    ax.axis(False)

    # Draw nodes
    x, y, z = np.array([pos[v] for v in sorted(G.nodes)]).T  # node coordinates
    ax.scatter(x, y, z, s=500)

    # Draw edges
    for u, v in G.edges():
        x, y, z = np.array([pos[u], pos[v]]).T
        ax.plot(x, y, z, 'gray', alpha=0.5)

    fig.tight_layout()
    plt.show()


def graphs_grid(graphs, labels_mask, cols=4, figsize=(16, 16)):
    N = len(graphs)
    fig, axes = plt.subplots(math.ceil(N/cols), cols, figsize=figsize)
    for i, ax in enumerate(axes.reshape(-1)):
        ax.axis(False)
        if i < N:
            graph = graphs[i]
            color = 'green' if labels_mask[i] else 'red'
            # G = to_networkx(graph, to_undirected=True)
            G = nx.DiGraph(graph)
            nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0), arrows=False, with_labels=False, node_size=150, node_color=color, width=0.8, ax=ax)
    plt.tight_layout()
    plt.show()
