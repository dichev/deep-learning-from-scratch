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


def n_grams(doc, n=3):
    words = doc.split()
    grams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return grams


def skip_grams(sequence, half_window=2, n=2, padding_token=0):
    grams, full_context = [], []
    sequence = [s for s in sequence if s != padding_token]  # drop the paddings
    if len(sequence) > 1:
        for i, word in enumerate(sequence):
            neighbours = sequence[max(0, i-half_window):i] + sequence[i+1:i+half_window+1]
            for comb in combinations(neighbours, n-1):
                grams.append([word, *comb])
                full_context.append(neighbours)

    return grams, full_context


class TextVocabulary:

    def __init__(self, sequences, max_vocab_size=None, padding='<PAD>', unknown='<UNK>'):
        words = [token for seq in sequences for token in seq]
        dictionary = [(padding, 0), (unknown, 0)] + Counter(words).most_common()  # [(word, counts)]

        if max_vocab_size:
            assert max_vocab_size > 2, 'The vocabulary size cannot be less than 2, because it must include the <padding> and <unknown> tokens'
            dictionary = dictionary[:max_vocab_size]

        self.size = len(dictionary)
        self.counts = np.array([counts for word, counts in dictionary])
        self.to_idx = {word: idx for idx, (word, counts) in enumerate(dictionary)}
        self.to_token = {idx: word for word, idx in self.to_idx.items()}

    def encode(self, sequence):
        return np.array([self.to_idx[token] if token in self.to_idx else 1 for token in sequence], dtype=int)

    def encode_batch(self, sequences, seq_length=10):
        encoded = np.zeros((len(sequences), seq_length), dtype=int)
        for i, seq in enumerate(sequences):  # not vectorized for readability
            encoded[i, :len(seq)] = self.encode(seq[:seq_length])
        return encoded

    def decode(self, tokens, sep=' '):
        return sep.join([self.to_token[idx] for idx in tokens if 0 < idx < self.size])

    def print_human(self, sequences):
        for seq in sequences:
            print(f'{seq} -> ', ' '.join([self.to_token[idx] for idx in seq if idx>0]))

    def __repr__(self):
        str = '\n Top 10 tokens:\n'
        for i in range(10):
            str += f' {i}: {self.to_token[i]} ({self.counts[i]})\n'
        return f'TextVocabulary(size={self.size})' + str