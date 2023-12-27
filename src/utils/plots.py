from matplotlib import pyplot as plt
import torch

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
    plt.suptitle(title)
    fig, axs = plt.subplots(N, 2, figsize=(3,N*1.3))
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
