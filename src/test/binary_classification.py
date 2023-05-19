import torch
from matplotlib import pyplot as plt
from lib import plots, optimizers
from data import data_gen
from perceptron import Perceptron


# Hyperparams
LEARN_RATE = 0.01
EPOCHS = 1000


# Define the model
model = Perceptron(input_size=2)
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

    print(f'#{i+1:<3} matched={torch.sum(model.predict(X)==y).item()}/{y.shape[0]}, cost={cost.item()} ')
    history.append((model.W.flatten().detach(), cost.item()))


# Plot the results
W, cost = zip(*history); W = torch.vstack(W)
plt.plot(range(EPOCHS), cost); plt.xscale('log'); plt.title('Loss'); plt.xlabel('iterations'); plt.show()
plt.scatter(W[:, 0], W[:, 1], s=2, alpha=0.5); plt.title('Parameters (weight) space'); plt.xlabel(f'$w_1$'); plt.ylabel(f'$w_2$'); plt.show()
plots.decision_boundary_2d(X, y, model.predict)
