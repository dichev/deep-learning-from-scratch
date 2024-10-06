import torch
from lib.functions.activations import sigmoid

class HopfieldNetwork:

    def __init__(self, n, device='cpu'):
        self.n = n
        self.W = torch.zeros(n, n, device=device, dtype=torch.float)

    def fit(self, X):
        assert X.shape == (len(X), self.n), f'Expected input to be a 2D array of shape (batch, n_features), but got {X.shape}'
        assert X.dtype is torch.float, f'Expected input to be a float tensor, but got {X.dtype}'

        # do an outer product sum for all the training samples as first step (Hebbian rule, no gradients)
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
        print(f'optimize the weights:')
        for i in range(epochs):
            print(f'epoch={i}, E={self.avg_energy(X):.2f}')

            samples = torch.randperm(len(X))
            for sample in samples:
                x = X[sample]
                not_x = self.forward(x, max_negative_steps)  # negative sampling for the nearest parasite patterns
                self.W += lr * (torch.outer(x, x) - torch.outer(not_x, not_x)).fill_diagonal_(0)

        print(f'done,    E={self.avg_energy(X):.2f}')

    def avg_energy(self, X):
        return torch.tensor([self.energy(x) for x in X]).mean()


class RestrictedBoltzmannMachine:

    def __init__(self, n_visible, n_hidden, device='cpu'):
        self.W = torch.randn(n_visible, n_hidden, device=device) * 0.01
        self.v_bias = torch.zeros(n_visible, device=device, dtype=torch.float)
        self.h_bias = torch.zeros(n_hidden, device=device, dtype=torch.float)

    def sample_h(self, v, as_probs=False):
        h = sigmoid(v @ self.W + self.h_bias)
        if not as_probs:
            h = torch.bernoulli(h)
        return h

    def sample_v(self, h, as_probs=False):
        v = sigmoid(h @ self.W.T + self.v_bias)
        if not as_probs:
            v = torch.bernoulli(v)
        return v

    def update(self, x, lr=0.1, k_reconstructions=1):
        B, N = x.shape

        # positive phase
        v = x
        h = self.sample_h(v, as_probs=True)
        pos_gradient = v.T @ h  # outer sum product (track correlations between v and h)

        # negative phase (contrastive divergence)
        h_k, v_k = h, v
        for i in range(k_reconstructions):  # additional k iterations
            v_k = self.sample_v(h_k, as_probs=True)
            if i < k_reconstructions - 1:
                h_k = self.sample_h(v_k, as_probs=False)
            else:  # last iteration must yield binary hidden states
                h_k = self.sample_h(v_k, as_probs=True)
        neg_gradient = v_k.T @ h_k  # outer sum product

        # update
        self.W += lr * (pos_gradient - neg_gradient) / B
        self.v_bias += lr * (v.mean(dim=0) - v_k.mean(dim=0))
        self.h_bias += lr * (h.mean(dim=0) - h_k.mean(dim=0))

    def sample(self, x=None, n_samples=1, burn_in=10):
        if x is not None:
            assert x.unique().tolist() == [0, 1], f'Expected input to be a binary tensor [0, 1], but got {x.unique().tolist()}'
            v = x
        else:
            probs = torch.full_like(self.v_bias, .5).reshape(1, -1)
            v = torch.bernoulli(probs)

        samples = []
        for i in range(burn_in + n_samples):  # Gibbs sampling
            h = self.sample_h(v)
            v = self.sample_v(h)
            if i >= burn_in:
                samples.append(v)

        return torch.stack(samples)

    def reconstruct(self, x):  # given corrupted input image, returns the expected probs (not binary states) of the encoded image
        assert x.unique().tolist() == [0, 1], f'Expected input to be a binary tensor [0, 1], but got {x.unique().tolist()}'
        v = x

        h = self.sample_h(v, as_probs=True)
        v = self.sample_v(h, as_probs=True)

        return v
