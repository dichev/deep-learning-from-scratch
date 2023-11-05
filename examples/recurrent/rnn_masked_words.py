import torch
from tqdm import trange

from models.recurrent_networks import RNN_factory
from preprocessing.text import TextVocabulary
from lib.functions.losses import cross_entropy
from lib.optimizers import Adam
from lib.training import batches

# training settings
EPOCHS = 100
BATCH_SIZE = 32
LEARN_RATE = 0.1
DEVICE = 'cuda'

# hyperparams
TIME_STEPS = 10
MAX_VOCAB_SIZE = 1000  # input/output size
HIDDEN_SIZE = 100

# Prepare text data
print('Data preprocessing..')
text = open('./data/deep-short.txt', 'r', encoding="utf-8").read()  # todo: cleanup the text
text = text.split()
vocab = TextVocabulary([text], MAX_VOCAB_SIZE, special_tokens=('<MASK>',))
print(vocab)
text_encoded = vocab.encode(text)
cut = len(text_encoded) % TIME_STEPS  # clip data to match the batch_size
X = torch.tensor(text_encoded[:-cut] if cut > 0 else text_encoded, dtype=torch.int64).reshape(-1, TIME_STEPS)


# Model # todo: compare forward vs backward vs bidirectional RNN
# net = RNN_factory(vocab.size, HIDDEN_SIZE, vocab.size, n_layers=1, direction='forward', device=DEVICE)
# net = RNN_factory(vocab.size, HIDDEN_SIZE, vocab.size, n_layers=3, direction='forward', device=DEVICE)
net = RNN_factory(vocab.size, HIDDEN_SIZE//2, vocab.size, n_layers=1, direction='bidirectional', device=DEVICE)
print(net.summary())
# plots.LaTeX(RNN, net.expression())
optim = Adam(net.parameters(), lr=LEARN_RATE)


# Train
N = len(X)
print(f'Fit {X.shape[0]} sequences (with {X.shape[1]} tokens each) into the model: {net}')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    loss = accuracy = 0
    for batch, batch_fraction in batches(X, batch_size=BATCH_SIZE, device=DEVICE):
        mask = torch.randint(1, TIME_STEPS-1, size=(len(batch), 1), device=DEVICE)
        x = batch.scatter(1, mask, vocab.to_idx['<MASK>'])
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
        print('\n# Test 5 sequences --------------------------------------------')
        for i in range(5):
            input, output, expected = [vocab.decode(v.detach().cpu().numpy()) for v in (x[i], y_hat[i][0].argmax(keepdims=True), y[i])]
            print(f"{'PASS' if expected==output else 'FAIL'} | " + input.replace('<MASK>', f'[{output}]'))
