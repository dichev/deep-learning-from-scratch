import torch
from matplotlib import pyplot as plt
from lib import plots, optimizers
from data import data_gen
from shallow_models import Perceptron, SVN, LeastSquareRegression, LogisticRegression


# Hyperparams
LEARN_RATE = 0.01
EPOCHS = 1000
plt.figure(figsize=(6, 10))

# Define the model
for Model in (Perceptron, SVN, LeastSquareRegression, LogisticRegression):
    model = Model(input_size=2)
    optimizer = optimizers.Optimizer(model.params, lr=LEARN_RATE)

    # Gather data
    X, y, data_plot = data_gen.linearly_separable()
    y = torch.where(y == 0, -1, 1)  # remap the target labels to be {-1, 1}, not {0, 1}
    N, D = X.shape

    # Fit the data
    history = []
    for i in range(EPOCHS):
        y_hat = model.forward(X)
        cost = model.cost(y, y_hat)
        cost.backward()
        optimizer.step().zero_grad()

        history.append((model.params[0].flatten().detach(), cost.item()))
        if i < 10 or i % 100 == 1 or i+1 == EPOCHS:
            print(f'#{i:<3} matched={torch.sum(model.predict(X)==y).item()}/{y.shape[0]}, cost={cost.item()} ')
        if i > 10 and abs(cost.item() - history[-10][1]) < 1e-5:
            break

    # Plot the results
    W, loss = zip(*history); W = torch.vstack(W)
    plt.subplot(311)
    plt.plot(range(len(loss)), loss, label=model.__class__.__name__); plt.xscale('log'); plt.title('Loss'); plt.xlabel('iterations'); plt.yscale('log'); plt.legend()
    plt.subplot(312)
    plt.scatter(W[:, 0], W[:, 1], s=2, alpha=0.5, label=model.__class__.__name__); plt.title('Parameters (weight) space'); plt.xlabel(f'$w_1$'); plt.ylabel(f'$w_2$'); plt.legend()
    plt.subplot(313)
    plots.decision_boundary_2d(X, y, model.predict)

plt.tight_layout()
plt.show()





