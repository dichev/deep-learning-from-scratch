import torch

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

