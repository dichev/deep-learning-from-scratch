import torch
from matplotlib import pyplot as plt
from sklearn import datasets
from lib import plots, optimizers
from shallow_models import Perceptron, SVM, LeastSquareRegression, LogisticRegression


# Hyperparams
LEARN_RATE = 0.01
EPOCHS = 1000
plt.figure(figsize=(6, 10))


# Gather data
n_samples, n_features, n_classes = N, D, C = 100, 2, 2
_X, _y = datasets.make_blobs(n_samples=N, n_features=D, centers=C, cluster_std=1.2, random_state=2)
# plt.scatter(_X[:, 0], _X[:, 1], c=_y, edgecolors='k');  plt.xlabel(f'$x_1$'); plt.ylabel(f'$x_2$'); plt.show()
X = torch.Tensor(_X)
y = torch.where(torch.Tensor(_y).long() == 0, -1, 1)


# Define the model
for Model in (Perceptron, SVM, LeastSquareRegression, LogisticRegression):
    model = Model(input_size=D)
    optimizer = optimizers.Optimizer(model.params, lr=LEARN_RATE)

    # Fit the data
    history = []
    for i in range(EPOCHS):
        y_hat = model.forward(X)
        cost = model.cost(y, y_hat.flatten())
        cost.backward()
        optimizer.step().zero_grad()

        history.append((model.params[0].flatten().detach().clone(), cost.item()))
        if i < 10 or i % 100 == 1 or i+1 == EPOCHS:
            print(f'#{i:<3} matched={torch.sum(model.predict(X)==y).item()}/{y.shape[0]}, cost={cost.item()} ')
        if i > 10 and abs(cost.item() - history[-10][1]) < 1e-5:
            break

    # Plot the results
    W, loss = zip(*history); W = torch.vstack(W)
    plt.subplot(311)
    plt.plot(range(len(loss)), loss, label=model.__class__.__name__); plt.xscale('log'); plt.title('Loss'); plt.xlabel('iterations'); plt.yscale('log'); plt.legend()
    plt.subplot(312)
    plt.scatter(W[:, 0], W[:, 1], s=2, alpha=0.5, label=model.__class__.__name__); plt.title('Parameters (weight) evolution'); plt.xlabel(f'$w_1$'); plt.ylabel(f'$w_2$'); plt.legend()
    plt.subplot(313)
    plots.decision_boundary_2d(X, y, model.predict)

plt.tight_layout()
plt.show()





