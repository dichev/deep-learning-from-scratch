import torch

def batches(X, y=None, batch_size=1024, shuffle=True, device=None):
    n = len(X)
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        batch = indices[i:i+batch_size]
        if y is None:
            yield X[batch].to(device), len(batch)/n
        else:
            yield X[batch].to(device), y[batch].to(device), len(batch)/n
