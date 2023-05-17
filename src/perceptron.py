import numpy as np
from matplotlib import pyplot as plt

from lib.activations import sign
from lib import data_gen, plots

# Hyperparams
LEARN_RATE = 0.01
EPOCHS = 1000


# Perceptron model
class Perceptron:
    def __init__(self, input_size):
        self.W = np.random.randn(1, input_size)  # (1,D)
        self.b = np.zeros((1, 1))                # (1,1)
        self._cache = None

    def forward(self, X):
        z = X @ self.W.T + self.b   # (N, D)x(D, 1) + (1, 1)  --> (N, 1)
        z = z.flatten()
        y_hat = sign(z)

        self._cache = {'X': X, 'z': z}
        return y_hat

    def backward(self, y, y_hat, lr=0.01):
        assert y.shape == y_hat.shape
        X, z = self._cache['X'], self._cache['z']

        cost = np.maximum(0, -y*z).sum()  # using a smooth loss (perceptron criterion)

        # grads
        I = (y != y_hat)  # will update the params only for the miss-classifications
        Y_missed = (I * y).reshape(1, -1)
        dW = -Y_missed @ X                 # (1,N) x (N, D)  -->  (1, D)
        db = -Y_missed.sum(keepdims=True)  # (1,1)

        # update
        self.W -= lr * dW
        self.b -= lr * db

        return cost


# Gather data
X, y, data_plot = data_gen.linearly_separable()
y = np.where(y == 0, -1, 1)  # remap the target labels to be {-1, 1}, not {0, 1}
N, D = X.shape


# Fit the data
model = Perceptron(input_size=D)
history = []
for i in range(EPOCHS):
    y_hat = model.forward(X)
    cost = model.backward(y, y_hat, lr=LEARN_RATE)

    print(f'#{i+1:<3} matched={np.sum(y==y_hat)}/{y.size}, {cost=} ')
    history.append((model.W.flatten(), cost))


# Plot the results
W, cost = zip(*history); W = np.array(W)
plt.plot(range(EPOCHS), cost); plt.xscale('log'); plt.title('Loss'); plt.xlabel('iterations'); plt.show()
plt.scatter(W[:, 0], W[:, 1], s=2, alpha=0.5); plt.title('Parameters (weight) space'); plt.xlabel(f'$w_1$'); plt.ylabel(f'$w_2$'); plt.show()
plots.decision_boundary_2d(X, y, model.forward)
