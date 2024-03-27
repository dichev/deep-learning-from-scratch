import torch
from lib.functions.activations import log_softmax


def entropy(p, logits=True):
    if logits:
        log_prob = log_softmax(p)
    else:
        log_prob = torch.log(p)

    losses = -(p * log_prob).sum(dim=-1)
    return losses.mean()


def cross_entropy(y_hat, y, logits=True, ignore_idx=None):
    if logits:
        log_prob = log_softmax(y_hat)
    else:
        log_prob = torch.log(y_hat)

    if y_hat.shape == y.shape:  # when y are one-hot or probabilities vectors
        losses = -(y * log_prob).sum(dim=-1)
    else:  # when y are indices, directly select the target class
        losses = -torch.gather(log_prob, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

    if ignore_idx is not None:
        mask = (y != ignore_idx)
        losses *= mask
        return losses.sum() / mask.sum()

    return losses.mean()
