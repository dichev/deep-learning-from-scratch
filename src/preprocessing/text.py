import re
from itertools import combinations
from collections import defaultdict, Counter

_patterns = (
    # (re.compile(r'[^a-zA-Z0-9_\-\s]'), ''),  # remove any special character
    # (re.compile(r'[\-_]'), ' '),             # convert dashes to spaces
    (re.compile(r'([,.!?])'), r' \1 '),        # insert space between punctuations
    (re.compile(r'\s+'), ' '),                 # normalize spaces
)

def clean_text(doc, lang='en'):
    doc = doc.lower()
    for pattern, repl in _patterns:
        doc = pattern.sub(repl, doc)
    return doc


def n_grams(seq, n=3):
    grams = [' '.join(seq[i:i+n]) for i in range(len(seq)-n+1)]
    return grams


def skip_grams(sequence, half_window=2, n=2, padding_token=0):
    grams, full_context = [], []
    sequence = [s for s in sequence if s != padding_token]  # drop the paddings
    if len(sequence) > 1:
        for i, word in enumerate(sequence):
            neighbors = sequence[max(0, i-half_window):i] + sequence[i+1:i+half_window+1]
            for comb in combinations(neighbors, n-1):
                grams.append([word, *comb])
                full_context.append(neighbors)

    return grams, full_context


def merge_tokens(tokens, pair, idx):
    i = 0
    new_tokens = []
    while i < len(tokens):
        a, b = tokens[i], tokens[i + 1] if i < len(tokens) - 1 else None
        if pair == (a, b):
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(a)
            i += 1
    return new_tokens


def byte_pair_encoding(words_freq: Counter | dict, num_merges=100, end_of_word_token='·'):
    """
    Paper: Neural Machine Translation of Rare Words with Subword Units
    https://arxiv.org/pdf/1508.07909.pdf
    """

    # Split all characters of the vocabulary words (and concat the end_of_word_token to the last char)
    words_freq = {tuple(word) + (end_of_word_token,): freq for word, freq in words_freq.items()}

    for m in range(num_merges):
        # Find most frequent pair (bigram)
        pairs = defaultdict(int)
        for subwords, freq in words_freq.items():
            if len(subwords) > 1:
                for first, second in zip(subwords, subwords[1:]):
                    pairs[(first, second)] += freq
        if not pairs:
            break

        # Get most frequent pair
        best = max(pairs, key=pairs.get)

        # Merge the best pair into the next symbol/integer
        words_freq = {tuple(merge_tokens(tokens, best, ''.join(best))): freq for tokens, freq in words_freq.items()}
        print(f'Merge {m + 1:>3}/{num_merges}: freq={pairs[best]} {best} -> {"".join(best)}')

    return words_freq


def byte_pair_encoding_byte_level(words_freq: Counter | dict, num_merges=100, start_idx=256):
    merges = {}
    for m in range(num_merges):
        # Find most frequent pair (bigram)
        pairs = defaultdict(int)
        for subwords, freq in words_freq.items():
            if len(subwords) > 1:
                for first, second in zip(subwords, subwords[1:]):
                    pairs[(first, second)] += freq
        if not pairs:
            break

        # Get most frequent pair
        best = max(pairs, key=pairs.get)

        # Merge the best pair into the next integer
        merges[best] = start_idx + m
        words_freq = {tuple(merge_tokens(tokens, best, start_idx + m)): freq for tokens, freq in words_freq.items()}
        print(f'BPE Merge {m + 1:>3}/{num_merges}: freq={pairs[best]} {best} -> {start_idx + m}')

    return words_freq, merges
