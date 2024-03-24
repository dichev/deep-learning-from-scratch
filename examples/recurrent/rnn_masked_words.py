import matplotlib.pyplot as plt
import torch
import math
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from models.recurrent_networks import SimpleRNN, LSTM, GRU, EchoStateNetwork, LangModel
from preprocessing.text import TextVocabulary
from lib.functions.losses import cross_entropy
from lib.optimizers import Adam
from lib.training import batches
from utils import rng
from lib.regularizers import grad_clip_norm_, L2_regularizer


# training settings
rng.seed_global(1)
EPOCHS = 500
BATCH_SIZE = 32
LEARN_RATE = 0.1
DEVICE = 'cuda'

# hyperparams
TIME_STEPS = 15
MAX_VOCAB_SIZE = 1000  # input/output size
HIDDEN_SIZE = 128

# Prepare text data
print('Data preprocessing..')
text = open('./data/deep-short.txt', 'r', encoding="utf-8").read()  # todo: cleanup the text
text = text.split()
vocab = TextVocabulary([text], MAX_VOCAB_SIZE, special_tokens=('<MASK>',))
print(vocab)
text_encoded = vocab.encode(text)
cut = len(text_encoded) % TIME_STEPS  # clip data to match the batch_size
X = torch.tensor(text_encoded[:-cut] if cut > 0 else text_encoded, dtype=torch.int64).reshape(-1, TIME_STEPS)

# Models
models = {  # todo: compare with similar size of parameters
    'RNN_1L':    LangModel(SimpleRNN(vocab.size, HIDDEN_SIZE, n_layers=1, direction='forward', layer_norm=False)),
    'RNN_1L LayerNorm':   LangModel(SimpleRNN(vocab.size, HIDDEN_SIZE, n_layers=1, direction='forward', layer_norm=True)),
    'RNN_2L LayerNorm':   LangModel(SimpleRNN(vocab.size, HIDDEN_SIZE, n_layers=2, direction='forward', layer_norm=True)),
    'BiRNN_1L LayerNorm': LangModel(SimpleRNN(vocab.size, HIDDEN_SIZE // 2, n_layers=1, direction='bidirectional', layer_norm=True)),

    'EchoState Sparse': EchoStateNetwork(vocab.size, HIDDEN_SIZE, vocab.size),

    'LSTM_1L':   LangModel(LSTM(vocab.size, HIDDEN_SIZE, n_layers=1, direction='forward')),
    'LSTM_2L':   LangModel(LSTM(vocab.size, HIDDEN_SIZE, n_layers=2, direction='forward')),
    'BiLSTM_1L': LangModel(LSTM(vocab.size, HIDDEN_SIZE // 2, n_layers=1, direction='bidirectional')),

    'GRU_1L':   LangModel(GRU(vocab.size, HIDDEN_SIZE, n_layers=1, direction='forward')),
    'GRU_2L':   LangModel(GRU(vocab.size, HIDDEN_SIZE, n_layers=2, direction='forward')),
    'BiGRU_1L': LangModel(GRU(vocab.size, HIDDEN_SIZE // 2, n_layers=1, direction='bidirectional')),
    'BiGRU_2L': LangModel(GRU(vocab.size, HIDDEN_SIZE // 2, n_layers=2, direction='bidirectional')),
}

for model_name, net in models.items():
    net.to(DEVICE)
    net.summary()
    # plots.LaTeX(RNN, net.expression())
    optimizer = Adam(net.parameters(), lr=LEARN_RATE)

    # Tracker
    now = datetime.now().strftime('%b%d %H-%M-%S')
    writer = SummaryWriter(f'runs/{model_name} GradClip T={TIME_STEPS} params={net.n_params} - {now}', flush_secs=2)

    # Training loop
    N = len(X)
    print(f'Fit {X.shape[0]} sequences (with {X.shape[1]} tokens each) into the model: {model_name}')
    pbar = trange(1, EPOCHS+1, desc='EPOCH')
    for epoch in pbar:
        loss = accuracy = grad_norm = 0
        for batch, batch_fraction in batches(X, batch_size=BATCH_SIZE, device=DEVICE):
            mask = torch.randint(1, TIME_STEPS-1, size=(len(batch), 1), device=DEVICE)
            x = batch.scatter(1, mask, vocab.to_idx['<MASK>'])
            y = batch.gather(1, mask)

            # Update
            optimizer.zero_grad()
            y_hat, _ = net.forward(x)
            y_hat = y_hat[torch.arange(len(batch)), mask.ravel(), :].unsqueeze(1)
            cost = cross_entropy(y_hat, y, logits=True)
            cost.backward()
            grad_clip_norm_(net.parameters(), 1.)
            optimizer.step()

            # Metrics
            grad_norm += net.grad_norm() * batch_fraction
            loss += cost.item() * batch_fraction
            accuracy += (y_hat.argmax(dim=-1) == y).sum().item()

        # Metrics
        writer.add_scalar('t/Loss', loss, epoch)
        writer.add_scalar('t/Perplexity', math.exp(loss), epoch)
        writer.add_scalar('t/Accuracy', accuracy/N, epoch)
        writer.add_scalar('a/Gradients Norm', grad_norm, epoch)
        writer.add_scalar('a/Weights Norm', net.weight_norm(), epoch)
        pbar.set_postfix(cost=f"{loss:.4f}", accuracy=f"{100*accuracy/N:.2f}%")

        if epoch == 1 or epoch % 10 == 0:
            print('\n# Test 5 sequences --------------------------------------------')
            for i in range(5):
                input, output, expected = [vocab.decode(v.detach().cpu().numpy()) for v in (x[i], y_hat[i][0].argmax(keepdims=True), y[i])]
                print(f"{'PASS' if expected==output else 'FAIL'} | " + input.replace('<MASK>', f'[{output}]'))

            for name, param in net.parameters():
                if 'bias' not in name:
                    writer.add_histogram('params/' + name.replace('.', '/'), param, epoch)
                    writer.add_histogram('grad/'+name.replace('.', '/'), param.grad, epoch)  # note this is a sample from the last mini-batch
