import numpy as np
from tqdm import tqdm as progress
from preprocessing.text import word_tokenizer
from utils.rng import pick_uniform
import preprocessing.text as text
from preprocessing.text import TextVocabulary


# settings
half_window = 2
n_negative_samples = 4
max_vocab_size = 4096
sequence_length = 10


def generate_training_batch(sequences, vocab_size):
    targets_, contexts_, labels_ = [], [], []  # the first dimension  will vary

    for sequence in progress(sequences):
        sequence = list(np.trim_zeros(sequence, 'b'))  # remove right padding
        if len(sequence) < 2:
            continue  # no grams for single word or no word

        # Generate positive skip-grams  # todo: exclude <unknown> words?
        skip_grams = text.skip_grams(sequence, half_window=half_window, n=2)

        # Sample negative skip-grams
        indices = np.arange(1, vocab_size)  # skips padding at 0
        negative_samples = []
        for target, context in skip_grams:
            window = np.arange(max(target-half_window, 0), min(target+half_window+1, vocab_size)) - 1  # skips padding at 0
            neg_samples = pick_uniform(indices, n_negative_samples, exclude=window)  # todo: in real applications sample from log uniform distribution over *ordered by frequency* vocabulary (Zipf's law)
            negative_samples.append(neg_samples)

        # Construct training batches
        n = len(skip_grams)
        batch = np.hstack((skip_grams, negative_samples))
        targets_.append(batch[:, 0])
        contexts_.append(batch[:, 1:])
        labels_.append(np.hstack((np.ones((n, 1)), np.zeros((n, n_negative_samples)))))

    return np.hstack(targets_), np.vstack(contexts_), np.vstack(labels_)


# Read the text documents
with open('../data/shakespeare.txt', 'r') as f:
    docs = [line.strip() for line in f if line.strip()]

# Tokenize and index
text_tokenized = [word_tokenizer(line) for line in docs]
vocab = TextVocabulary(text_tokenized, max_vocab_size=max_vocab_size)
text_encoded = vocab.encode(text_tokenized)

# Prepare training batch
targets, contexts, labels = generate_training_batch(text_encoded, vocab.size)
vocab.print_human(text_encoded[:5])
print('\ntargets:', targets.shape, '\ncontexts:', contexts.shape, '\nlabels:', labels.shape, '\n')

