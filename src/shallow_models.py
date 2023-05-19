import torch
from lib.activations import sigmoid


class Linear:
    def __init__(self, input_size):
        self.W = torch.zeros(input_size, 1, requires_grad=True)  # (D,1)
        self.b = torch.zeros(1, 1, requires_grad=True)           # (1,1)
        self.params = (self.W, self.b)

    def forward(self, X):
        z = X @ self.W + self.b    # (N, D)x(D, 1) + (1, 1)  --> (N, 1)
        z = z.flatten()
        return z


class Perceptron:
    def __init__(self, input_size):
        self.linear = Linear(input_size)
        self.params = self.linear.params

    def forward(self, X):
        return self.linear.forward(X)

    def cost(self, y, y_hat):  # perceptron_criterion
        assert torch.all((y == -1) | (y == 1))

        loss = -y * y_hat  # using a smooth loss (perceptron criterion)
        loss[loss < 0] = 0.
        return loss.sum()

    @torch.no_grad()
    def predict(self, X):
        z = self.linear.forward(X)
        y_hat = torch.sign(z)
        return y_hat

class SVN(Perceptron):
    def cost(self, y, y_hat):  # perceptron_criterion
        assert torch.all((y == -1) | (y == 1))

        loss = 1 - y * y_hat
        loss[loss < 0] = 0.
        return loss.sum()

class LeastSquareRegression(Perceptron):
    def cost(self, y, y_hat):  # perceptron_criterion
        assert torch.all((y == -1) | (y == 1))

        losses = (y-y_hat)**2
        return losses.mean()


class LogisticRegression:
    def __init__(self, input_size):
        self.linear = Linear(input_size)
        self.params = self.linear.params

    def forward(self, X):
        z = self.linear.forward(X)
        z = sigmoid(z)
        return z

    def cost(self, y, y_hat):
        assert torch.all((y == -1) | (y == 1)) and torch.all((y_hat >= 0) & (y_hat <= 1))

        losses = -torch.log(torch.abs(y/2 - .5 + y_hat))
        return losses.sum()

    @torch.no_grad()
    def predict(self, X, threshold=.5):
        y_hat = self.forward(X)
        return torch.where(y_hat > threshold, 1, -1)
