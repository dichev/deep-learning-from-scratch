import torch

def batches(X, y, batch_size, shuffle=True, device=None):
    n = len(y)
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        batch = indices[i:i+batch_size]
        yield X[batch].to(device), y[batch].to(device), len(batch)/n
