import torch
import torch.nn.functional as F
from lib.activations import sigmoid, softmax


class Linear:
    def __init__(self, input_size, output_size=1):
        self.W = torch.rand(input_size, output_size, requires_grad=True)  # (D, C)
        self.b = torch.rand(1, output_size, requires_grad=True)           # (1, C)
        self.params = (self.W, self.b)

    def forward(self, X):
        z = X @ self.W + self.b    # (N, D)x(D, C) + (1, C)  --> (N, C)
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
        z = self.linear.forward(X).flatten()
        y_hat = torch.sign(z).long()
        return y_hat

class SVM(Perceptron):
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
        p = self.linear.forward(X).flatten()
        y_hat = torch.where(p > threshold, 1, -1).long()
        return y_hat


class MulticlassPerceptron:
    def __init__(self, input_size, output_size):
        self.linear = Linear(input_size, output_size)
        self.params = self.linear.params
        self.output_size = output_size

    def forward(self, X):
        Z = self.linear.forward(X)
        return Z

    def cost(self, y, y_hat):
        y_hat_correct = (y_hat*y).sum(dim=1).view(-1, 1)  # y is one-hot vector

        margin = (y_hat - y_hat_correct)
        margin[margin < 0] = 0.

        isMisclassified = (torch.argmax(y_hat, dim=1) != torch.argmax(y, dim=1)).view(-1, 1)
        isMaximum = F.one_hot(margin.argmax(dim=1), num_classes=self.output_size)

        loss = margin * isMisclassified * isMaximum
        return loss.sum()

    @torch.no_grad()
    def predict(self, X):
        P = self.forward(X)
        return torch.argmax(P, dim=-1)


class MulticlassSVM(MulticlassPerceptron):

    def cost(self, y, y_hat):
        y_hat_correct = (y_hat*y).sum(dim=1).view(-1, 1)  # y is one-hot vector

        isMisclassified = (torch.argmax(y_hat, dim=1) != torch.argmax(y, dim=1)).view(-1, 1)

        loss = (y_hat - y_hat_correct + 1) * isMisclassified
        loss[loss < 0] = 0.

        return loss.sum()


class MultinomialLogisticRegression:
    def __init__(self, input_size, output_size):
        self.linear = Linear(input_size, output_size)
        self.params = self.linear.params
        self.output_size = output_size

    def forward(self, X):
        Z = self.linear.forward(X)
        P = softmax(Z)
        return P

    def cost(self, y, y_hat):
        # cross-entropy (y is one-hot vector)
        losses = -y*torch.log(y_hat)  # (N,C) * (N,P) -> (N,L)
        return losses.sum()

    @torch.no_grad()
    def predict(self, X):
        P = self.forward(X)
        return torch.argmax(P, dim=-1)

