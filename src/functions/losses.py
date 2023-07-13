import torch
from functions.activations import log_softmax


def cross_entropy(y_hat, y, logits=True):  # equal to the relative entropy (D_KL) for one-hot labels
    if logits:
        log_prob = log_softmax(y_hat)
    else:
        log_prob = torch.log(y_hat)

    losses = -(y * log_prob).sum(dim=-1)
    return losses.mean()

