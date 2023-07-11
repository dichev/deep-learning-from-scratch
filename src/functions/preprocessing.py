import torch


def index_mappings(labels): # todo rename index_tokens
    if hasattr(labels, 'unique'):  # pandas series, torch tensors
        unique = labels.unique()
    else:
        unique = list(dict.fromkeys(labels))  # note set(labels) can't preserve the order

    vocab = {cls: idx for idx, cls in enumerate(unique)}
    vocab_inverse = {idx: cls for cls, idx in vocab.items()}
    return vocab, vocab_inverse


def integer_encoder(labels):
    vocab, vocab_inverse = index_mappings(labels) # todo tmp
    encoded = torch.tensor([vocab[d] for d in labels], dtype=torch.long)
    return encoded, vocab, vocab_inverse

integer_encoder(['cat', 'dog', 'fish', 'dog', ..., 'cow']) # => [0, 1, 2, 1, ..., n]
