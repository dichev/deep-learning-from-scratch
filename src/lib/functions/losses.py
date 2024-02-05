import torch
from lib.functions.activations import log_softmax
from collections import Counter

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

def evaluate_accuracy_per_class(y_hat, y, classes):
    predicted, actual = y_hat.argmax(1), y.argmax(1)
    correct = (predicted == actual)
    overall_accuracy = correct.float().mean().item()

    all = Counter(actual.tolist())
    matched = Counter(actual[predicted != actual].tolist())

    accuracy_per_class = {classes[idx]: matched[idx] / all[idx] for idx in sorted(all.keys())}
    return overall_accuracy, accuracy_per_class

def accuracy(y_hat, y):
    return ((y_hat == y).sum() / len(y)).item()
