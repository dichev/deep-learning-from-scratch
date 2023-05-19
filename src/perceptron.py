import torch
from matplotlib import pyplot as plt

from lib import plots, optimizers
from data import data_gen

# Hyperparams
LEARN_RATE = 0.01
EPOCHS = 1000


# Perceptron model
class Perceptron:

    def __init__(self, input_size):
        self.W = torch.randn(input_size, 1, requires_grad=True)  # (1,D)
        self.b = torch.zeros(1, 1, requires_grad=True)           # (1,1)
        self.params = (self.W, self.b)

    def forward(self, X):
        z = X @ self.W + self.b    # (N, D)x(D, 1) + (1, 1)  --> (N, 1)
        z = z.flatten()
        return z

    def cost(self, y, y_hat):
        return torch.maximum(torch.zeros_like(y), -y * y_hat).sum()  # using a smooth loss (perceptron criterion)

    @torch.no_grad()
    def predict(self, X):
        z = self.forward(X)
        return torch.sign(z)



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
    z = model.forward(X)
    cost = model.cost(y, z)
    cost.backward()
    optimizer.step().zero_grad()

    print(f'#{i+1:<3} matched={torch.sum(y==model.predict(X)).item()}/{y.shape[0]}, cost={cost.item()} ')
    history.append((model.W.flatten().detach(), cost.item()))


# Plot the results
W, cost = zip(*history); W = torch.vstack(W)
plt.plot(range(EPOCHS), cost); plt.xscale('log'); plt.title('Loss'); plt.xlabel('iterations'); plt.show()
plt.scatter(W[:, 0], W[:, 1], s=2, alpha=0.5); plt.title('Parameters (weight) space'); plt.xlabel(f'$w_1$'); plt.ylabel(f'$w_2$'); plt.show()
plots.decision_boundary_2d(X, y, model.predict)
