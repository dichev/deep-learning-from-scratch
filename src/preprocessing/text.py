import re
from itertools import combinations
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


