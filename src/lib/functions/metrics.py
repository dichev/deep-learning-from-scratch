import math
from collections import Counter
import torch
from preprocessing.text import n_grams


def accuracy(y_hat, y, ignore_idx=None):
    correct = (y_hat == y).float()

    if ignore_idx is not None:
        mask = (y != ignore_idx)
        return ((correct * mask).sum() / mask.sum()).item()

    return correct.mean().item()


def evaluate_accuracy_per_class(y_hat, y, classes):
    predicted, actual = y_hat.argmax(1), y
    correct = (predicted == actual)
    overall_accuracy = correct.float().mean().item()

    all = Counter(actual.tolist())
    matched = Counter(actual[predicted != actual].tolist())

    accuracy_per_class = {classes[idx]: matched[idx] / all[idx] for idx in sorted(all.keys())}
    return overall_accuracy, accuracy_per_class



def BLEU(y_hat, y, max_n=4):
    if len(y_hat) == 0:
        return 0.

    # compute precisions of all the n-grams
    p = torch.zeros(max_n)
    for i in range(min(max_n, len(y))):
        k = i+1
        y_grams = Counter(n_grams(y, k))
        y_hat_grams = Counter(n_grams(y_hat, k))

        matches = sum(min(count, y_grams[gram]) for gram, count in y_hat_grams.items())
        total_pred = sum(y_hat_grams.values())

        p[i] = matches / total_pred if total_pred > 0 else 0
        # print(k, matches, total_pred, p[i].item(), y_grams, y_hat_grams)

    # penalizes shorter predicted sequences, because they tend to yield a higher scores
    penalty = math.exp(min(0, 1 - len(y)/len(y_hat)))

    # assign a greater weight to a longer n-gram, since matching longer n-grams is more difficult
    n = torch.arange(max_n) + 1
    weights = (1/2**n)

    # compute the BLEU scores (on log scale)
    score = penalty * torch.exp((weights * torch.log(p)).sum(dim=-1))
    return score.item()

