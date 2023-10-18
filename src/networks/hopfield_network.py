import torch

class HopfieldNetwork:

    def __init__(self, n, device='cpu'):
        self.n = n
        self.W = torch.zeros(n, n, device=device, dtype=torch.float)

    def fit(self, X):
        assert X.shape == (len(X), self.n), f'Expected input to be a 2D array of shape (batch, n_features), but got {X.shape}'
        assert X.dtype is torch.float, f'Expected input to be a float tensor, but got {X.dtype}'

        # do an outer product sum for all the training samples as first step
        self.W += (X.T @ X).fill_diagonal_(0)  # removing the self connections on the diagonal

    def forward(self, x, max_steps=100):
        energy = self.energy(x)
        for i in range(max_steps):
            x = torch.sign(self.W @ x)

            # iterate until there is no change in the energy (or alternatively until x converge to stable values)
            energy_delta = self.energy(x) - energy
            energy += energy_delta
            # print(f'#{i+1}/{max_steps} ΔE: {energy_delta}')
            if energy_delta == 0:
                break

        return x

    def energy(self, y):  # energy = sum of the products of each neuron with its field at given time
        return -y @ (self.W @ y) / 2 / len(y)


class HopfieldNetworkOptimized(HopfieldNetwork):
    """
     Minimize the energy for the target patterns AND maximize it for all other (parasite) patterns
        W = argmin sum E(y) - sum E(y')        , where y' are the nearest parasite patterns
    """

    def fit(self, X, lr=1., epochs=5, max_negative_steps=2):
        super().fit(X)  # first do an outer product sum for all the training samples as first step

        # then iterate over the target and parasite patterns until convergence
        for i in range(epochs):
            print(f'epoch={i}, E={self.avg_energy(X):.2f}')

            samples = torch.randperm(len(X))
            for sample in samples:
                x = X[sample]
                not_x = self.forward(x, max_negative_steps)  # negative sampling for the nearest parasite patterns
                self.W += lr * (torch.outer(x, x) - torch.outer(not_x, not_x)).fill_diagonal_(0)

        print(f'done, E={self.avg_energy(X):.2f}')

    def avg_energy(self, X):
        return torch.tensor([self.energy(x) for x in X]).mean()
