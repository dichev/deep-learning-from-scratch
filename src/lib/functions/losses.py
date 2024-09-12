import torch
from lib.functions.activations import log_softmax
from preprocessing.integer import one_hot, label_smooth

def entropy(p, logits=True):
    if logits:
        log_prob = log_softmax(p)
    else:
        log_prob = torch.log(p)

    losses = -(p * log_prob).sum(dim=-1)
    return losses.mean()


def cross_entropy(y_hat, y, logits=True, ignore_idx=None, label_smoothing=0.0):
    y_is_indices = y_hat.shape != y.shape

    # Ignore some tokens (like the padding)
    if ignore_idx is not None:
        ignore_mask = (y if y_is_indices else y.argmax(dim=-1)) != ignore_idx

    # Label smoothing
    if label_smoothing:
        if y_is_indices:
            y = one_hot(y, num_classes=y_hat.shape[-1])
            y_is_indices = False
        y = label_smooth(y, eps=label_smoothing)

    # Compute the total loss
    log_prob = log_softmax(y_hat) if logits else torch.log(y_hat)
    if y_is_indices:  # directly select the target class
        losses = -torch.gather(log_prob, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    else:
        losses = -(y * log_prob).sum(dim=-1)

    # Ignore some tokens (like the padding)
    if ignore_idx is not None:
        return (losses * ignore_mask).sum() / ignore_mask.sum()

    return losses.mean()


def mse_loss(y_hat, y, dim=None):
    e = y_hat - y
    mse = (e**2).mean(dim=dim)
    return mse