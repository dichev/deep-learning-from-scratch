import torch
from models.layers import Linear, Embedding


class MatrixFactorization:
    def __init__(self, n_users, n_animes, rank, device='cpu'):
        self.U = torch.randn(n_users, rank, device=device, requires_grad=True)   # (user, k)
        self.V = torch.randn(n_animes, rank, device=device, requires_grad=True)  # (anime, k)
        self.params = (self.U, self.V)

    def forward(self, user, anime):
        # dot only along the known ratings in the data, not over all like U @ V.T
        R_hat = (self.U[user] * self.V[anime]).sum(axis=1)     # sum[ (N, k)*(N, k) ]  --> (N)
        return R_hat

    @torch.no_grad()
    def predict(self, user, anime):
        predictions = self.U[user] @ self.V[anime].T
        return predictions.detach().cpu()


class AutoencoderLinear:
    def __init__(self, input_size, hidden_size, output_size, device='cpu'):
        self.encoder = Linear(input_size, hidden_size, device)
        self.decoder = Linear(hidden_size, output_size, device)
        self.params = (self.encoder.params, self.decoder.params)

    def forward(self, X):
        U = self.encoder.forward(X)
        V = self.decoder.forward(U)
        return V


class Word2Vec:
    def __init__(self, vocab_size, embedding_size, device='cpu'):
        self.target = Embedding(vocab_size, embedding_size, device=device)
        self.context = Embedding(vocab_size, embedding_size, device=device)
        self.params = (self.target.params, self.context.params)

    def forward(self, targets, contexts):
        target_vectors = self.target.forward(targets)       # (B, 1) -> (B, embed)
        context_vectors = self.context.forward(contexts)    # (B, context) -> (B, context, embed)
        z = torch.einsum('be,bce->bc', target_vectors, context_vectors)  # (B, E)  @  (B, C, E).T  -> (B, C)
        return z  # logits

