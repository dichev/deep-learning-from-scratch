import torch
from lib.functions.activations import log_softmax

def cross_entropy(y_hat, y, logits=True):
    if logits:
        log_prob = log_softmax(y_hat)
    else:
        log_prob = torch.log(y_hat)

    if y_hat.shape == y.shape:  # when y are one-hot or probabilities vectors
        losses = -(y * log_prob).sum(dim=-1)
    else:  # when y are indices, directly select the target class
        losses = -torch.gather(log_prob, dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

    return losses.mean()

@torch.no_grad()
def evaluate_accuracy(y_hat, y):
    predicted, actual = y_hat.argmax(1), y.argmax(1)
    correct = (predicted == actual)
    return correct.float().mean().item()
