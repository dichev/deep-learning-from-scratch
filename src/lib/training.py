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


def batched_text(data, batch_size, seq_len, device=None):
    n = len(data) - seq_len
    epoch_steps = n // seq_len // batch_size + 1
    for _ in range(epoch_steps):
        indices = torch.randint(n, size=(batch_size, ))
        x = torch.stack([data[i:i + seq_len] for i in indices])
        y = torch.stack([data[i+1:i + seq_len + 1] for i in indices])
        yield x.to(device), y.to(device)
