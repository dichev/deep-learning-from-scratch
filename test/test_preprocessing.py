import pytest
import torch
import pandas as pd

from preprocessing.floats import normalizeMinMax
from preprocessing.integer import index_encoder
from preprocessing.text import clean_text, n_grams, skip_grams, byte_pair_encoding
from preprocessing.vocab import TextVocabulary, BPETokenizer
from preprocessing.integer import one_hot, label_smooth


def test_normalizeMinMax():
    # Test with a tensor of positive integers
    x = torch.tensor([1, 2, 3, 4, 5])
    y = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.all(normalizeMinMax(x) == y)

    # Test with a tensor of negative integers
    x = torch.tensor([-5, -4, -3, -2, -1])
    y = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    assert torch.all(normalizeMinMax(x) == y)

    # Test with a mix of positive and negative integers
    x = torch.tensor([-5, 0, 5])
    y = torch.tensor([0.0, 0.5, 1.0])
    assert torch.all(normalizeMinMax(x) == y)

    # Test with a tensor of zeros and non-zero number
    a = torch.Tensor([[0., 0., 0.], [5., 5., 5.]])
    b = torch.Tensor([[0., 0., 0.], [1., 1., 1.]])
    assert torch.all(normalizeMinMax(a) == b)


def test_index_encoder_with_python_list():
    labels = ['cat', 'dog', 'fish', 'dog', 'cow']
    encoded, vocab, vocab_inverse = index_encoder(labels)
    assert torch.all(encoded == torch.tensor([0, 1, 2, 1, 3]))
    assert vocab == {'cat': 0, 'dog': 1, 'fish': 2, 'cow': 3}
    assert vocab_inverse == {0: 'cat', 1: 'dog', 2: 'fish', 3: 'cow'}

def test_index_encoder_with_pandas_series():
    labels = pd.Series(['x', 'y', 'x', 'z', 'y'])
    encoded, vocab, vocab_inverse = index_encoder(labels)
    assert torch.all(encoded == torch.tensor([0, 1, 0, 2, 1]))
    assert vocab == {'x': 0, 'y': 1, 'z': 2}
    assert vocab_inverse == {0: 'x', 1: 'y', 2: 'z'}

def test_n_grams():
    doc = 'the wide road shimmered in the hot sun'
    expected = ['the wide', 'wide road', 'road shimmered', 'shimmered in', 'in the', 'the hot', 'hot sun']
    assert n_grams(doc.split(), n=2) == expected

def test_skip_grams_with_tokens():
    seq = 'the wide road shimmered'.split()
    expected_grams = [['the', 'wide'], ['the', 'road'], ['wide', 'the'], ['wide', 'road'], ['wide', 'shimmered'], ['road', 'the'], ['road', 'wide'], ['road', 'shimmered'], ['shimmered', 'wide'], ['shimmered', 'road']]
    expected_full_context = [['wide', 'road'], ['wide', 'road'], ['the', 'road', 'shimmered'], ['the', 'road', 'shimmered'], ['the', 'road', 'shimmered'], ['the', 'wide', 'shimmered'], ['the', 'wide', 'shimmered'], ['the', 'wide', 'shimmered'], ['wide', 'road'], ['wide', 'road']]

    grams, full_context = skip_grams(seq, half_window=2, n=2, padding_token='\n')
    assert grams == expected_grams
    assert full_context == expected_full_context

def test_skip_grams_with_indices():
    seq = [0, 1, 0, 0, 4, 0]
    expected_grams = [[1, 4], [4, 1]]
    expected_full_context = [[4], [1]]

    grams, full_context = skip_grams(seq, half_window=2, n=2, padding_token=0)
    assert grams == expected_grams
    assert full_context == expected_full_context

def test_text_vocabulary():
    docs = [
        'Welcome to the AI world!',
        'The wide road shimmered in the hot hot hot hot sun.',
    ]
    docs_tokenized = [clean_text(doc).split() for doc in docs]
    vocab = TextVocabulary(docs_tokenized, max_vocab_size=100)

    assert torch.all(vocab.encode_batch([clean_text('Welcome to the AI world!').split()], seq_length=10) == torch.tensor([[4, 5, 3, 6, 7, 8, 0, 0, 0, 0]]))
    assert vocab.decode([4, 5, 3, 6, 7, 0, 0]) == 'welcome to the ai world'

    sequences = vocab.encode_batch(docs_tokenized, seq_length=10)
    vocab.print_human(sequences)

    assert torch.all(sequences == torch.tensor([[ 4,  5,  3,  6,  7,  8,  0,  0,  0,  0], [ 3,  9, 10, 11, 12,  3,  2,  2,  2,  2]]))
    assert vocab.to_token == {0: '<PAD>', 1: '<UNK>', 2: 'hot', 3: 'the', 4: 'welcome', 5: 'to', 6: 'ai', 7: 'world', 8: '!', 9: 'wide', 10: 'road', 11: 'shimmered', 12: 'in', 13: 'sun', 14: '.'}
    assert vocab.to_idx == {'<PAD>': 0, '<UNK>': 1, 'hot': 2, 'the': 3, 'welcome': 4, 'to': 5, 'ai': 6, 'world': 7, '!': 8, 'wide': 9, 'road': 10, 'shimmered': 11, 'in': 12, 'sun': 13, '.': 14}


def test_byte_pair_encoding():
    # example from the original paper:
    words = {'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}
    expected = {'low·': 5, 'low e r ·': 2, 'newest·': 6, 'wi d est·': 3}

    vocab = byte_pair_encoding(words, num_merges=10, end_of_word_token='·')
    actual = {' '.join(tokens): freq for tokens, freq in vocab.items()}
    assert actual == expected


def test_BPETokenizer():
    text = ('Firstly, we initialize the symbol vocabulary with the character vocabulary, '
            'and represent each word as a sequence of characters, plus a special end-of-word symbol, '
            'which allows us to restore the original tokenization after translation. '
            'We iteratively count all symbol pairs and replace each occurrence of the most frequent pair (‘A’, ‘B’) with a new symbol ‘AB’. '
            'Each merge operation produces a new symbol which represents a character n-gram. '
            'Frequent character n-grams (or whole words) are eventually merged into a single symbol, thus BPE requires no shortlist. '
            'The final symbol vocabulary size is equal to the size of the initial vocabulary, '
            'plus the number of merge operations – the latter is the only hyperparameter of the algorithm.'
            'For efficiency, we do not consider pairs that cross word boundaries. '
            'The algorithm can thus be run on the dictionary extracted from a text, with each word being weighted by its frequency.')

    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=20)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.decode(tokenizer.encode('unseen words: Здравейте 🤚')) == 'unseen words: Здравейте 🤚'
    assert len(tokenizer.vocab) == 256 + 20


def test_label_smoothing():
    expected = torch.tensor([[0.025, 0.925, 0.025, 0.025]])
    actual = label_smooth(torch.tensor([[0, 1, 0, 0]]), eps=0.1)
    assert torch.allclose(expected, actual)

    batch_size, num_classes = 20, 10
    y = torch.randint(num_classes, (batch_size, 1))
    y_hot = one_hot(y, num_classes)
    y_soft = label_smooth(y_hot, eps=0.1)
    assert not torch.any(y_soft == 0.) and not torch.any(y_soft == 1.)
    assert torch.allclose(y_soft.sum(dim=-1), torch.ones(num_classes))


