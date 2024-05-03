import torch
from collections import Counter
from preprocessing.text import byte_pair_encoding
from abc import abstractmethod


class Vocab:
    def __init__(self, sequences, padding_token="<PAD>", unknown_token='<UNK>', special_tokens=(), **kwargs):
        self.padding_token = padding_token
        self.unknown_token = unknown_token
        self.special_tokens = special_tokens

        vocab = self.create_vocab(sequences)

        self.size = len(vocab)
        self.counts = [counts for word, counts in vocab]
        self.to_idx = {word: idx for idx, (word, counts) in enumerate(vocab)}
        self.to_token = {idx: word for word, idx in self.to_idx.items()}

    @abstractmethod
    def create_vocab(self, sequences):
        pass

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



class TextVocabulary(Vocab):

    def __init__(self, sequences, max_vocab_size=None, min_freq=None, padding_token="<PAD>", unknown_token='<UNK>', special_tokens=()):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        super().__init__(sequences, padding_token, unknown_token, special_tokens)


    def create_vocab(self, sequences):
        min_vocab_size = len(self.special_tokens) + 2  # including padding and unknown tokens
        assert self.max_vocab_size is None or self.max_vocab_size > min_vocab_size, f'The vocabulary size cannot be less than {min_vocab_size}, because it must include the special tokens'

        words = [token for seq in sequences for token in seq]
        limit = (self.max_vocab_size - min_vocab_size) if self.max_vocab_size is not None else None

        selected = Counter(words).most_common(limit)
        if self.min_freq is not None:
            selected = [(word, freq) for word, freq in selected if freq >= self.min_freq]

        count_unknown = len(words) - sum(counts for word, counts in selected)
        vocab = [(self.padding_token, 0), (self.unknown_token, count_unknown)] + [(token, 0) for token in self.special_tokens] + selected
        return vocab

    def encode(self, sequence: list|str) -> list:
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


class BytePairVocabulary(Vocab):
    def __init__(self, sequences, num_merges, end_of_word_token='<EOW>',  padding_token="<PAD>", unknown_token='<UNK>', special_tokens=()):
        self.num_merges = num_merges
        self.end_of_word_token = end_of_word_token
        super().__init__(sequences, padding_token, unknown_token, special_tokens)

    def create_vocab(self, sequences):
        words = dict(Counter(token for seq in sequences for token in seq))
        vocab_joint = byte_pair_encoding(words, self.num_merges, self.end_of_word_token)

        # Generate the new vocabulary
        vocab = Counter()
        for words, freq in vocab_joint.items():
            vocab.update({token: freq for token in words})

        vocab = [(self.padding_token, 0), (self.unknown_token, 0)] + [(token, 0) for token in self.special_tokens] + vocab.most_common()
        return vocab

    def encode(self, sequence: str):
        encoded = []
        for word in sequence.split(' '):
            word += self.end_of_word_token
            while len(word):
                for i in reversed(range(1, len(word)+1)):
                    if word[:i] in self.to_idx:
                        encoded.append(self.to_idx[word[:i]])
                        word = word[i:]
                        break
                else:
                    encoded.append(self.to_idx[self.unknown_token])
                    word = word[1:]

        return encoded

    def decode(self, tokens: list, trim_after='<PAD>', sep=' '):
        trim_idx = self.to_idx[trim_after] if trim_after in self.to_idx else -1
        if trim_idx in tokens:
            pos = tokens.index(trim_idx)
            tokens = tokens[:pos]

        return ''.join([self.to_token[idx].replace(self.end_of_word_token, sep) for idx in tokens]).rstrip()
