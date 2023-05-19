from matplotlib import pyplot as plt
import torch

def decision_boundary_2d(X, Y, classifier):
    xx, yy = torch.meshgrid(torch.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                            torch.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))

    Z = classifier(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=-1))
    Z = Z.view(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap='Blues')
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap='Blues')
    plt.xlabel(f'$x_1$')
    plt.ylabel(f'$x_2$')
    plt.title('Decision boundary')
    plt.show()