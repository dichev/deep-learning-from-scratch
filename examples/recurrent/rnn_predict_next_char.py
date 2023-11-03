import torch
from tqdm import trange

from networks.recurrent_networks import UniRNN
from preprocessing.text import TextVocabulary
from functions.losses import cross_entropy
from models.optimizers import SGD, Adam
from models.training import batches
from utils import plots

# training settings
EPOCHS = 200
BATCH_SIZE = 2048
LEARN_RATE = 0.1
DEVICE = 'cuda'

# hyperparams
TIME_STEPS = 15
HIDDEN_SIZE = 100


# Prepare text data
print('Data preprocessing..')
text = open('./data/deep.txt', 'r', encoding="utf-8").read()  # todo: cleanup the text
vocab = TextVocabulary(list(text))
print(vocab)
text_encoded = vocab.encode(text)
cut = len(text_encoded) % TIME_STEPS  # clip data to match the batch_size
X = torch.tensor(text_encoded[:-cut], dtype=torch.int64).reshape(-1, TIME_STEPS)


# Model
net = UniRNN(vocab.size, HIDDEN_SIZE, vocab.size, device=DEVICE)
optim = Adam(net.parameters(), lr=LEARN_RATE)
print(net)
plots.LaTeX('RNN', net.expression())


# Train
N = len(X)
print(f'Fit {X.shape[0]} sequences (with {X.shape[1]} tokens each) into the model: {net}')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    loss = 0
    for batch, batch_fraction in batches(X, batch_size=BATCH_SIZE, device=DEVICE):
        x, y = batch[:, :-1], batch[:, 1:]  # note the first token is not predicted, while the last token is used only as target
        y_hat, _ = net.forward(x, logits=True)
        cost = cross_entropy(y_hat, y, logits=True)
        cost.backward()
        optim.step().zero_grad()
        loss += cost.item() * batch_fraction

    pbar.set_postfix(cost=f"{loss:.4f}")
    if epoch == 1 or epoch % 10 == 0:
        print('\n# Sampling --------------------------------------------')
        print('-> [The simplest neural network] ' + vocab.decode(net.sample(30, temperature=.5, seed_seq=vocab.encode('The simplest neural network')), sep=''))
        print('-> [W ⇐ W +] ' + vocab.decode(net.sample(30, temperature=.5, seed_seq=vocab.encode('W ⇐ W +')), sep=''))
        print('-> [random char] ' + vocab.decode(net.sample(30, temperature=.5), sep=''))

