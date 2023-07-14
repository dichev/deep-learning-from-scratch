import re
from itertools import combinations
import numpy as np  # todo switch to torch
from collections import Counter

_patterns = (
    (re.compile(r'[^a-zA-Z0-9_\-\s]'), ''),  # remove any special character
    (re.compile(r'[\-_]'), ' '),             # convert dashes to spaces
    (re.compile(r'\s+'), ' '),               # normalize spaces
)

def word_tokenizer(doc, split=True):
    doc = doc.lower()
    for pattern, repl in _patterns:
        doc = pattern.sub(repl, doc)
    return doc.split() if split else doc

# word_tokenizer("Hello- @$#A?I    w%%orld \n!") # -> ['hello', 'ai', 'world']


def n_grams(doc, n=3):
    words = doc.split()
    grams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return grams

# n_grams('the wide road shimmered in the hot sun', n=2)  # -> [['the wide'], ['wide road'], ..]


def skip_grams(sequence, half_window=2, n=2):
    assert type(sequence) is list
    grams, full_context = [], []
    for i in range(len(sequence)):
        word, neighbours = sequence[i], sequence[max(0, i-half_window):i] + sequence[i+1:i+half_window+1]
        for comb in combinations(neighbours, n - 1):
            grams.append([word, *comb])
            full_context.append(neighbours)

    return grams, full_context

# grams, full_context = skip_grams('the wide road shimmered in the hot sun'.split(), half_window=2, n=2)   # -> [..., ['wide, the'], ['wide, road'], ['wide, shimmered'] ..]
# grams, full_context = skip_grams([0, 1, 2, 3, 4, 0, 5, 6], half_window=2, n=2)



class TextVocabulary:

    def __init__(self, sequences, max_vocab_size=None, padding='<PAD>', unknown='<UNK>'):
        words = [token for seq in sequences for token in seq]
        dictionary = [padding, unknown]
        dictionary += [word for word, freq in Counter(words).most_common()]

        if max_vocab_size:
            assert max_vocab_size > 2, 'The vocabulary size cannot be less than 2, because it must include the <padding> and <unknown> tokens'
            dictionary = dictionary[:max_vocab_size]

        self.size = len(dictionary)
        self.to_idx = {word: idx for idx, word in enumerate(dictionary)}
        self.to_token = {idx: word for word, idx in self.to_idx.items()}

    def encode(self, sequences, seq_length=10):
        encoded = np.zeros((len(sequences), seq_length), dtype=int)
        for i, seq in enumerate(sequences):  # not vectorized for readability
            encoded[i, :len(seq)] = [self.to_idx[token] if token in self.to_idx else 1 for token in seq[:seq_length]]
        return encoded

    def translate(self, tokens):
        return ' '.join([self.to_token[idx] for idx in tokens if 0 < idx < self.size])

    def print_human(self, sequences):
        for seq in sequences:
            print(f'{seq} -> ', ' '.join([self.to_token[idx] for idx in seq if idx>0]))


# docs = [
#     'Welcome to the AI world!',
#     'The wide road shimmered in the hot hot hot hot sun.',
# ]
# docs_tokenized = [word_tokenizer(doc) for doc in docs]
# vocab = TextVocabulary(docs_tokenized, max_vocab_size=100)
# vocab.encode([word_tokenizer('Welcome to the AI world!')])
# vocab.translate([4, 5, 3, 6, 7, 0, 0])
# sequences = vocab.encode(docs_tokenized, seq_length=10)
# vocab.print_human(sequences)
# print(vocab.to_token)










