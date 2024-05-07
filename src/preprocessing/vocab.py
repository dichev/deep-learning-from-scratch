import torch
from collections import Counter
from preprocessing.text import byte_pair_encoding_byte_level, merge_tokens
from abc import abstractmethod
import regex as rex

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


class BPETokenizer:  # Byte pair encoding (on byte level)

    def __init__(self):
        self.merges = None
        self.vocab = None
        self.cache = None

        # using GPT4 split pattern
        contractions = r"'(?i:[sdmt]|ll|ve|re)"        # case-insensitive and language specific (similar to "'s|'t|'re|'ve|'m|'ll|'d|"
        word = r"[^\r\n\p{L}\p{N}]?+\p{L}+"            # unicode letters (words) not starting with space (or starting with any character except newlines, letters, and numbers)
        number = r"\p{N}{1,3}"                         # numbers up to 3 digits
        punctuation = r" ?[^\s\p{L}\p{N}]++[\r\n]*"    # punctuations, not space, letter or number
        whitespace = r"\s*[\r\n]|\s+(?!\S)|\s+"        # the whitespaces left
        self.pat = rex.compile(f"{contractions}|{word}|{number}|{punctuation}|{whitespace}")

    def train(self, corpus: str, num_merges):
        words = rex.findall(self.pat, corpus)                # split text into words, numbers and other
        words = [self.to_byte_int(word) for word in words]   # convert to byte ints and count
        words_freq = Counter(words)                          # use a dictionary of word frequencies for efficient BPE
        _, merges = byte_pair_encoding_byte_level(words_freq, num_merges, start_idx=256)
        self.load(merges)

    def load(self, merges):
        assert sorted(merges.values()) == list(merges.values()) and len(set(merges.values())) == len(merges), f'The merges must be sorted and unique, but got {merges}'

        self.cache = {}
        self.merges = merges
        self.vocab = {idx: bytes([idx]) for idx in range(256)}  # initialize with all single bytes
        for (a, b), idx in merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]

    def decode(self, tokens):
        return self.bytes_to_text([self.vocab[idx] for idx in tokens])

    def encode(self, text: str):
        words = rex.findall(self.pat, text)    # split text into words, numbers and other
        encoded = [token for word in words for token in self.encode_word(word)]
        return encoded

    def encode_word(self, word: str):
        if word in self.cache:
            return self.cache[word]

        tokens = self.to_byte_int(word)  # each byte as integer in the range [0, 255]
        if len(tokens) > 1:
            for pair, idx in self.merges.items():  # important - merges must be sorted (which is asserted by the load method)
                tokens = merge_tokens(tokens, pair, idx)

        self.cache[word] = tokens
        return tokens

    @staticmethod
    def to_byte_int(word: str):
        return tuple(map(int, word.encode('utf-8')))

    @staticmethod
    def bytes_to_text(seq: list[bytes]):
        return b"".join(seq).decode('utf-8', errors='replace')
