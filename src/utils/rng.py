import numpy as np
import torch
import random

def seed_global(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pick_uniform(arr, n=1, exclude=None):
    if exclude is not None:
        mask = np.ones(len(arr), dtype=bool)
        mask[exclude] = False
        arr = arr[mask]
    return np.random.choice(arr, n)

def sample_from(probs, n=1, exclude=None):
    if exclude is not None:
        probs = probs.copy()
        probs[exclude] = 0.
        probs = probs / probs.sum()
    return np.random.choice(len(probs), n, p=probs)
