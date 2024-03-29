import torch
from lib.layers import Module, Linear, Embedding
from lib.functions import init
from lib.base import Param


class MatrixFactorization(Module):
    def __init__(self, n_users, n_animes, rank):
        self.U = Param((n_users, rank))   # (user, k)
        self.V = Param((n_animes, rank))  # (anime, k)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.U.normal_()
        self.V.normal_()

    def forward(self, user, anime):
        # dot only along the known ratings in the data, not over all like U @ V.T
        R_hat = (self.U[user] * self.V[anime]).sum(axis=1)     # sum[ (N, k)*(N, k) ]  --> (N)
        return R_hat

    @torch.no_grad()
    def predict(self, user, anime):
        predictions = self.U[user] @ self.V[anime].T
        return predictions.detach().cpu()


class AutoencoderLinear(Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.encoder = Linear(input_size, hidden_size)
        self.decoder = Linear(hidden_size, output_size)

    def forward(self, X):
        U = self.encoder.forward(X)
        V = self.decoder.forward(U)
        return V


class Word2Vec(Module):
    """
    Paper: Efficient Estimation of Word Representations in Vector Space
    https://arxiv.org/pdf/1301.3781.pdf
    """
    def __init__(self, vocab_size, embedding_size):
        self.target = Embedding(vocab_size, embedding_size)
        self.context = Embedding(vocab_size, embedding_size)

    def forward(self, targets, contexts):
        target_vectors = self.target.forward(targets)       # (B, 1) -> (B, embed)
        context_vectors = self.context.forward(contexts)    # (B, context) -> (B, context, embed)
        z = torch.einsum('be,bce->bc', target_vectors, context_vectors)  # (B, E)  @  (B, C, E).T  -> (B, C)
        return z  # logits

