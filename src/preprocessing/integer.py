import torch


def index_encoder(labels):
    """
        Encodes tokens to indices
        ['cat', 'dog', 'fish', 'dog', ..., 'cow']) => [0, 1, 2, 1, ..., n]
    """
    if hasattr(labels, 'unique'):  # pandas series, torch tensors
        unique = labels.unique()
    else:
        unique = list(dict.fromkeys(labels))  # note set(labels) can't preserve the order

    vocab = {cls: idx for idx, cls in enumerate(unique)}
    vocab_inverse = {idx: cls for cls, idx in vocab.items()}
    encoded = torch.tensor([vocab[d] for d in labels], dtype=torch.long)
    return encoded, vocab, vocab_inverse
