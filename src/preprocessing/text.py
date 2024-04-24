import re
from itertools import combinations
from collections import defaultdict
import contractions


_patterns = (
    # (re.compile(r'[^a-zA-Z0-9_\-\s]'), ''),  # remove any special character
    # (re.compile(r'[\-_]'), ' '),             # convert dashes to spaces
    (re.compile(r'([,.!?])'), r' \1 '),        # insert space between punctuations
    (re.compile(r'\s+'), ' '),                 # normalize spaces
)

def clean_text(doc, lang='en'):
    doc = doc.lower()
    if lang == 'en':
        doc = contractions.fix(doc)
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


def merge_pairs(token, pair):
    a, b = pair
    token = ' '.join(token).replace(a + ' ' + b, a + b)
    return tuple(token.split())


def byte_pair_encoding(vocab, num_merges=100, end_of_word_token='·'):
    """
    Paper: Neural Machine Translation of Rare Words with Subword Units
    https://arxiv.org/pdf/1508.07909.pdf
    """

    # Split all characters of the vocabulary words (and concat the end_of_word_token to the last char)
    vocab = {tuple(word) + (end_of_word_token,): freq for word, freq in vocab.items()}

    for m in range(num_merges):
        # Count frequency of all "byte" pairs
        pairs = defaultdict(int)
        for subwords, freq in vocab.items():
            if len(subwords) > 1:
                for first, second in zip(subwords, subwords[1:]):
                    pairs[(first, second)] += freq

        if not pairs: break

        # Get most frequent pair
        best = max(pairs, key=pairs.get)

        # Merge byte pairs to a single symbol
        vocab = {merge_pairs(token, best): freq for token, freq in vocab.items()}
        # print(f'Merge {m + 1:>3}/{num_merges}: freq={pairs[best]} {best} -> {''.join(best)}')

    return vocab


