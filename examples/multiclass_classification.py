import torch
from matplotlib import pyplot as plt
from sklearn import datasets
from models import optimizers
from utils import plots
from models.shallow_models import MulticlassPerceptron, MulticlassSVM, MultinomialLogisticRegression
from preprocessing.integer import one_hot

# Hyperparams
LEARN_RATE = 0.005
EPOCHS = 1000
plt.figure(figsize=(6, 12))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Gather data
n_samples, n_features, n_classes = N, D, C = 100, 2, 5
_X, _y = datasets.make_blobs(n_samples=N, n_features=D, centers=C, cluster_std=0.6, random_state=2)
# plt.scatter(_X[:, 0], _X[:, 1], c=_y, edgecolors='k');  plt.xlabel(f'$x_1$'); plt.ylabel(f'$x_2$'); plt.show()
X = torch.Tensor(_X)
Y = one_hot(torch.Tensor(_y).long())


# Define the model
idx = 0
for Model in (MulticlassPerceptron, MulticlassSVM, MultinomialLogisticRegression):
    model = Model(input_size=D, output_size=C)
    optimizer = optimizers.SGD(model.parameters, lr=LEARN_RATE)

    # Fit the data
    history = []
    for i in range(EPOCHS):
        Y_hat = model.forward(X)
        cost = model.cost(Y, Y_hat)
        cost.backward()
        optimizer.step().zero_grad()

        history.append((tuple(model.parameters(named=False))[0].flatten().detach().clone(), tuple(model.parameters(named=False))[1].flatten().detach().clone(), cost.item()))
        if i < 10 or i % 100 == 1 or i+1 == EPOCHS:
            print(f'#{i:<3} matched={torch.sum(model.predict(X)==torch.argmax(Y, dim=-1)).item()}/{Y_hat.shape[0]}, cost={cost.item()} ')
        if i > 10 and abs(cost.item() - history[-10][-1]) < 1e-5:
            break

    # Plot the results
    W, b, loss = zip(*history); W = torch.vstack(W); b = torch.vstack(b)
    plt.subplot(411)
    plt.plot(range(len(loss)), loss, c=colors[idx], label=model.__class__.__name__); plt.xscale('log'); plt.title('Loss'); plt.xlabel('iterations'); plt.yscale('log'); plt.legend()
    plt.subplot(412)
    plt.plot((W**2).sum(axis=1).sqrt(), c=colors[idx]); plt.plot((b**2).sum(axis=1).sqrt(), c=colors[idx], alpha=.5); plt.title('Parameters L2 norm evolution'); plt.xlabel(f'iterations'); plt.ylabel(f'w'); plt.xscale('log');
    plt.subplot(413)
    plt.plot(W, c=colors[idx]); plt.title('Parameters (weight) evolution'); plt.xlabel(f'iterations'); plt.ylabel(f'w'); plt.xscale('log')
    plt.subplot(414)
    plots.decision_boundary_2d(X, torch.argmax(Y, dim=-1), model.predict)
    idx += 1

plt.tight_layout()
plt.show()
