import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm as progress

from preprocessing.text import word_tokenizer, TextVocabulary
import preprocessing.text as text
from utils.rng import pick_uniform
from functions.losses import cross_entropy
from models.optimizers import Optimizer
from models.autoencoders import Word2Vec


# data settings
half_window = 2
n_negative_samples = 4
max_vocab_size = 4096
sequence_length = 10
# training settings
EPOCH = 100
BATCH_SIZE = 10240
LEARN_RATE = 0.5
WORD_EMBEDDINGS_DIM = 10
DEVICE = 'cuda'


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
        negative_samples = np.array(negative_samples)

        # Construct training batches
        n = len(skip_grams)
        batch = torch.hstack((torch.tensor(skip_grams), torch.tensor(negative_samples)))
        targets_.append(batch[:, 0])
        contexts_.append(batch[:, 1:])
        labels_.append(torch.hstack((torch.ones((n, 1)), torch.zeros((n, n_negative_samples)))))

    return torch.hstack(targets_), torch.vstack(contexts_), torch.vstack(labels_)


# Read the text documents
print('Data preprocessing..')
with open('../data/shakespeare.txt', 'r') as f:
    docs = [line.strip() for line in f if line.strip()]

# Tokenize and index
text_tokenized = [word_tokenizer(line) for line in docs]
vocab = TextVocabulary(text_tokenized, max_vocab_size=max_vocab_size)
text_encoded = vocab.encode(text_tokenized)
vocab.print_human(text_encoded[:5])

# Prepare training batch
data = generate_training_batch(text_encoded, vocab.size)  # todo: doubled tensors
targets, contexts, labels = [torch.tensor(d, device=DEVICE) for d in data]
print('\ntargets:', targets.shape, '\ncontexts:', contexts.shape, '\nlabels:', labels.shape, '\n')

# Train a Word2Vec model
word2vec = Word2Vec(vocab.size, WORD_EMBEDDINGS_DIM, device=DEVICE)
optimizer = Optimizer(word2vec.params, lr=LEARN_RATE)

print('Fit word2vec model..')
N = len(targets)
history = []
for epoch in range(1, EPOCH):
    pbar = trange(1, N // BATCH_SIZE + 1, desc=f'EPOCH #{epoch}/{EPOCH}')
    accuracy, loss = 0., 0.
    for i in pbar:
        batch = torch.randint(0, N, (BATCH_SIZE,))
        y = labels[batch]
        y_hat_logit = word2vec.forward(targets[batch], contexts[batch])

        cost = cross_entropy(y_hat_logit, y, logits=True)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss += cost.item()
        predicted, actual = y_hat_logit.argmax(1), y.argmax(1)
        accuracy += (predicted == actual).float().mean().item()
        pbar.set_postfix(cost=loss/i, accuracy=accuracy/i)

    history.append((loss/pbar.total, accuracy/pbar.total))


# Plot the loss function
loss, accuracy = zip(*history)
plt.plot(range(len(loss)), loss); plt.title('Loss'); plt.xlabel('iterations'); plt.show()
