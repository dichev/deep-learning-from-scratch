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


def one_hot(x, num_classes=None):
    I = torch.eye(num_classes or x.max() + 1, device=x.device)
    return I[x]


def label_smooth(y_hot, eps=0.1):  # Reference: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf
    num_classes = y_hot.shape[-1]
    U = 1 / num_classes  # uniform prior
    return (1-eps)*y_hot + eps * U




