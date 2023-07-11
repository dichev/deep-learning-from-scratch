import numpy as np

def pick_uniform(arr, n, exclude=None):
    if exclude is not None:
        mask = np.ones(len(arr), dtype=bool)
        mask[exclude] = False
        arr = arr[mask]
    return np.random.choice(arr, n)

