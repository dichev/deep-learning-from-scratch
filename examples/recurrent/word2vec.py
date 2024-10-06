import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm as progress

from preprocessing.text import clean_text
from preprocessing.vocab import TextVocabulary
import preprocessing.text as text
from utils.rng import sample_from
from lib.functions.losses import cross_entropy
from lib.optimizers import SGD
from lib.autoencoders import Word2Vec


# data settings
half_window = 2
n_negative_samples = 4
max_vocab_size = 4096
sequence_length = 15
# training settings
EPOCHS = 1000
BATCH_SIZE = 1024
LEARN_RATE = 0.5
WORD_EMBEDDINGS_DIM = 10
DEVICE = 'cuda'


def word_sampling_table(word_counts, sampling_factor=1e-05):
    frequencies = word_counts / word_counts.sum()
    probs = np.zeros_like(frequencies)
    mask = frequencies != 0
    probs[mask] = np.sqrt(sampling_factor / frequencies[mask])
    probs[probs > 1] = 1.
    return probs


def generate_training_batch(sequences, word_counts, subsampling=True):
    targets_, contexts_, labels_ = [], [], []  # the first dimension  will vary

    # set counters UNK and PAD token counter to zero, to be not subsampled
    word_counts = np.array(word_counts)
    word_counts[:2] = 0

    # Generate distribution for the negative sampling (the factor 3/4 is empirically recommended in the word2vec paper)
    word_frequencies = word_counts ** (3/4) / np.sum(word_counts ** (3/4))

    # Subsampling - i.e. downsampling the more frequent words
    # Note the padding and unknown tokens have zero prob
    if subsampling:
        sampling_table = word_sampling_table(word_counts)
        filter = sampling_table[sequences] > np.random.random(sequences.shape)
        sequences = sequences * filter
        sequences = sequences[np.count_nonzero(sequences, axis=1) > 1]  # remove all sequences with less than 2 sampled tokens

    for sequence in progress(sequences):

        # Generate positive skip-grams
        skip_grams, full_context = text.skip_grams(sequence, half_window=half_window, n=2, padding_token=0)

        if skip_grams:
            # Sample negative skip-grams
            negative_samples = []
            for full_context_ in full_context:
                exclude = [0] + full_context_  # skips also the <padding> at 0
                # neg_samples = pick_uniform(np.arange(vocab_size), n_negative_samples, exclude=exclude)
                neg_samples = sample_from(word_frequencies, n_negative_samples, exclude=exclude)
                negative_samples.append(neg_samples)
            negative_samples = np.array(negative_samples)

            # Construct training batches
            n = len(skip_grams)
            batch = np.hstack((skip_grams, negative_samples))
            targets_.append(batch[:, 0])
            contexts_.append(batch[:, 1:])
            labels_.append(np.hstack((np.ones((n, 1)), np.zeros((n, n_negative_samples)))))

    return np.hstack(targets_), np.vstack(contexts_), np.vstack(labels_)


# Read the text documents
print('Data preprocessing..')
with open('./data/shakespeare.txt', 'r') as f:
    docs = [line.strip() for line in f if line.strip()]

# Tokenize and index
text_tokenized = [clean_text(line).split() for line in docs]
vocab = TextVocabulary(text_tokenized, max_vocab_size=max_vocab_size)
text_encoded = vocab.encode_batch(text_tokenized, seq_length=sequence_length).numpy()
vocab.print_human(text_encoded[:5])

# Prepare training batch
data = generate_training_batch(text_encoded, vocab.counts, subsampling=True)
targets, contexts, labels = [torch.tensor(d, device=DEVICE) for d in data]
print(f'\ntargets: {tuple(targets.shape)} | contexts: {tuple(contexts.shape)} | labels: {tuple(labels.shape)} \n')
assert torch.all(targets > 1) and torch.all(contexts > 1), 'The training data contains paddings or unknown words'

# Train a Word2Vec model
word2vec = Word2Vec(vocab.size, WORD_EMBEDDINGS_DIM).to(DEVICE)
optimizer = SGD(word2vec.parameters(), lr=LEARN_RATE)

N = len(targets)
history = []
print(f'Fit {N} training samples in word2vec model..')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    accuracy, loss = [], []
    indices = torch.randperm(N)
    for i in range(0, N, BATCH_SIZE):
        batch = indices[i:i+BATCH_SIZE]
        y = labels[batch]

        optimizer.zero_grad()
        y_hat_logit = word2vec.forward(targets[batch], contexts[batch])
        cost = cross_entropy(y_hat_logit, y, logits=True)
        cost.backward()
        optimizer.step()

        loss.append(cost.item())
        predicted, actual = y_hat_logit.argmax(1), y.argmax(1)
        accuracy.append((predicted == actual).float().mean().item())

    epoch_loss, epoch_accuracy = np.mean(loss), np.mean(accuracy)  # todo: not accurate for batches less than BATCH_SIZE
    pbar.set_postfix(cost=epoch_loss, accuracy=epoch_accuracy)
    history.append((epoch_loss, epoch_accuracy))


# Plot the loss function
loss, accuracy = zip(*history)
plt.plot(range(len(loss)), loss, label=f'loss = {loss[-1]:.2f}')
plt.plot(range(len(accuracy)), accuracy, label=f'accuracy = {accuracy[-1]:.2f}')
plt.title('Loss & accuracy'); plt.xlabel('iterations'); plt.legend(); plt.show()
