import torch
from collections import Counter


class TextVocabulary:

    def __init__(self, sequences, max_vocab_size=None, min_freq=None, padding_token="<PAD>", unknown_token='<UNK>', special_tokens=()):
        min_vocab_size = len(special_tokens) + 2  # including padding and unknown tokens
        assert max_vocab_size is None or max_vocab_size > min_vocab_size, f'The vocabulary size cannot be less than {min_vocab_size}, because it must include the special tokens'

        words = [token for seq in sequences for token in seq]
        limit = (max_vocab_size - min_vocab_size) if max_vocab_size is not None else None

        selected = Counter(words).most_common(limit)
        if min_freq is not None:
            selected = [(word, freq) for word, freq in selected if freq >= min_freq]

        count_unknown = len(words) - sum(counts for word, counts in selected)
        dictionary = [(padding_token, 0), (unknown_token, count_unknown)] + [(token, 0) for token in special_tokens] + selected

        self.size = len(dictionary)
        self.counts = [counts for word, counts in dictionary]
        self.to_idx = {word: idx for idx, (word, counts) in enumerate(dictionary)}
        self.to_token = {idx: word for word, idx in self.to_idx.items()}

    def encode(self, sequence: list) -> list:
        return [self.to_idx[token] if token in self.to_idx else 1 for token in sequence]

    def encode_batch(self, sequences: list[list[str]], seq_length=10):
        encoded = torch.zeros(len(sequences), seq_length, dtype=torch.long)
        for i, seq in enumerate(sequences):  # not vectorized for readability
            encoded[i, :len(seq)] = torch.tensor(self.encode(seq[:seq_length]))
        return encoded

    def decode(self, tokens: list, trim_after='<PAD>', sep=' ') -> str:
        trim_idx = self.to_idx[trim_after] if trim_after in self.to_idx else -1
        if trim_idx in tokens:
            pos = tokens.index(trim_idx)
            tokens = tokens[:pos]

        return sep.join([self.to_token[idx] for idx in tokens])

    def print_human(self, sequences):
        sequences = sequences.tolist() if isinstance(sequences, torch.Tensor) else sequences
        for seq in sequences:
            print(f'{seq} -> ', ' '.join([self.to_token[idx] for idx in seq if idx>0]))

    def __repr__(self):
        n = min(self.size, 10)
        tokens = f'\n Top {n} tokens:\n'
        for i in range(n):
            tokens += f' {i}: {self.to_token[i]} ({self.counts[i]})\n'

        return f'TextVocabulary(size={self.size})' + tokens
