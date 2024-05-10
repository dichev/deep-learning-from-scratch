import math
import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from models.recurrent_networks import SimpleRNN, LSTM, GRU, LangModel
from preprocessing.vocab import TextVocabulary
from lib.functions.losses import cross_entropy
from lib.optimizers import Adam
from lib.regularizers import L2_regularizer, grad_clip_norm_
from lib.training import batches
from utils import rng
from utils import plots

# training settings
rng.seed_global(1)
EPOCHS = 200
BATCH_SIZE = 1024
LEARN_RATE = 0.01
DEVICE = 'cuda'

# hyperparams
TIME_STEPS = 20
HIDDEN_SIZE = 150


# Prepare text data
print('Data preprocessing..')
text = open('./data/deep.txt', 'r', encoding="utf-8").read()  # todo: cleanup the text
vocab = TextVocabulary(list(text))
print(vocab)
text_encoded = vocab.encode(list(text))
cut = len(text_encoded) % TIME_STEPS  # clip data to match the batch_size
X = torch.tensor(text_encoded[:-cut], dtype=torch.long).reshape(-1, TIME_STEPS)


# Model
models = {
    'RNN LayerNorm': LangModel(SimpleRNN(vocab.size, HIDDEN_SIZE, n_layers=1, direction='forward', layer_norm=True)),
    'LSTM':          LangModel(LSTM(vocab.size, HIDDEN_SIZE, n_layers=1, direction='forward')),
    'GRU':           LangModel(GRU(vocab.size, HIDDEN_SIZE, n_layers=1, direction='forward')),
}

for model_name, net in models.items():
    net.to(DEVICE)
    net.summary()
    optimizer = Adam(net.parameters(), lr=LEARN_RATE)

    # Tracker
    now = datetime.now().strftime('%b%d %H-%M-%S')
    writer = SummaryWriter(f'runs/{model_name} T={TIME_STEPS} params={net.n_params} - {now}', flush_secs=2)

    # Train
    N = len(X)
    print(f'Fit {X.shape[0]} sequences (with {X.shape[1]} tokens each) into the model: {model_name}')
    pbar = trange(1, EPOCHS+1, desc='EPOCH')
    for epoch in pbar:
        loss = grad_norm = 0
        for batch, batch_fraction in batches(X, batch_size=BATCH_SIZE, device=DEVICE):
            x, y = batch[:, :-1], batch[:, 1:]  # note the first token is not predicted, while the last token is used only as target

            optimizer.zero_grad()
            y_hat, _ = net.forward(x)
            cost = cross_entropy(y_hat, y, logits=True)
            cost.backward()
            # grad_clip_norm_(net.parameters(), 1.)
            optimizer.step()

            # Metrics
            grad_norm += net.grad_norm() * batch_fraction
            loss += cost.item() * batch_fraction

        # Metrics
        writer.add_scalar('t/Loss', loss, epoch)
        writer.add_scalar('t/Perplexity', math.exp(loss), epoch)
        writer.add_scalar('a/Gradients Norm', grad_norm, epoch)
        writer.add_scalar('a/Weights Norm', net.weight_norm(), epoch)

        if epoch == 1 or epoch % 10 == 0:
            print('\n# Sampling --------------------------------------------')
            print('-> [The simplest neural network] ' + vocab.decode(net.sample(30, temperature=.5, seed_seq=vocab.encode('The simplest neural network')), sep=''))
            print('-> [W ⇐ W +] ' + vocab.decode(net.sample(30, temperature=.5, seed_seq=vocab.encode('W ⇐ W +')), sep=''))
            print('-> [random char] ' + vocab.decode(net.sample(150, temperature=.5), sep=''))

            for name, param in net.parameters():
                if 'bias' not in name:
                    writer.add_histogram('params/' + name.replace('.', '/'), param, epoch)
                    writer.add_histogram('grad/' + name.replace('.', '/'), param.grad, epoch)  # note this is a sample from the last mini-batch


