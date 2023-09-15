import torch


def data_split(X, y, sizes: list | tuple, shuffle=True):
    n = len(X)
    assert n == len(y), f"The data size {n} doesn't match to the targets size {len(y)}"

    # if fractions are provided, calculated the number of samples
    if sum(sizes) == 1.:
        sizes = [int(n * s) for s in sizes]
        sizes[0] += n - sum(sizes)
    assert sum(sizes) == n, f'Expected {n} samples, but got {sum(sizes)}'

    # shuffle data
    if shuffle:
        shuffled = torch.randperm(n)
        X, y = X[shuffled], y[shuffled]

    # split data
    sets = tuple()
    start = 0
    for end in sizes:
        sets += X[start:start + end], y[start:start + end]
        start += end

    return sets
