import torch

def integer_encoder(labels):
    if hasattr(labels, 'unique'):  # pandas series, torch tensors
        unique = labels.unique()
    else:
        unique = list(dict.fromkeys(labels))  # note set(labels) can't preserve the order

    key_to_idx = {cls: idx for idx, cls in enumerate(unique)}
    idx_to_key = {idx: cls for cls, idx in key_to_idx.items()}
    encoded = torch.tensor([key_to_idx[d] for d in labels], dtype=torch.long)
    return encoded, key_to_idx, idx_to_key

# integer_encoder(['cat', 'dog', 'fish', 'dog', ..., 'cow']) # => [0, 1, 2, 1, ..., n]