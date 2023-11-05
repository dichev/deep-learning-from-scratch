import torch
from tqdm import trange

from networks.recurrent_networks import UniRNN, BiRNN
from preprocessing.text import TextVocabulary
from functions.losses import cross_entropy
from models.optimizers import Adam
from models.training import batches
from utils import plots


# training settings
EPOCHS = 100
BATCH_SIZE = 32
LEARN_RATE = 0.1
DEVICE = 'cuda'

# hyperparams
TIME_STEPS = 10
HIDDEN_SIZE = 100

# Prepare text data
print('Data preprocessing..')
text = open('./data/shakespeare.txt', 'r', encoding="utf-8").read()[:30_000]
text = text.split()

vocab = TextVocabulary([text])
print(vocab)
text_encoded = vocab.encode(text)
cut = len(text_encoded) % TIME_STEPS  # clip data to match the batch_size
X = torch.tensor(text_encoded[:-cut] if cut > 0 else text_encoded, dtype=torch.int64).reshape(-1, TIME_STEPS)


# Model
# net = UniRNN(vocab.size, HIDDEN_SIZE, vocab.size, device=DEVICE)
net = BiRNN(vocab.size, HIDDEN_SIZE//2, vocab.size, device=DEVICE)
optim = Adam(net.parameters(), lr=LEARN_RATE)
print(net)
# plots.LaTeX('RNN', net.expression())


# Train
N = len(X)
print(f'Fit {X.shape[0]} sequences (with {X.shape[1]} tokens each) into the model: {net}')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    loss = accuracy = 0
    for batch, batch_fraction in batches(X, batch_size=BATCH_SIZE, device=DEVICE):
        mask = torch.randint(1, TIME_STEPS-1, size=(len(batch), 1), device=DEVICE)
        UNK_TOKEN = 1
        x = batch.scatter(1, mask, UNK_TOKEN)
        y = batch.gather(1, mask)

        y_hat, _ = net.forward(x, logits=True)
        y_hat = y_hat[torch.arange(len(batch)), mask.ravel(), :].unsqueeze(1)
        cost = cross_entropy(y_hat, y, logits=True)
        cost.backward()
        optim.step().zero_grad()

        loss += cost.item() * batch_fraction
        accuracy += (y_hat.argmax(dim=-1) == y).sum().item()

    pbar.set_postfix(cost=f"{loss:.4f}", accuracy=f"{100*accuracy/N:.2f}%")

    if epoch == 1 or epoch % 10 == 0:
        print('\n# Test 5 masked sequences --------------------------------------------')
        for i in range(5):
            input, output, expected = [vocab.decode(v.detach().cpu().numpy()) for v in (x[i], y_hat[i][0].argmax(keepdims=True), y[i])]
            print(f"{'PASS' if expected==output else 'Fail'} | " + input.replace('<UNK>', f'[{output}]'))
