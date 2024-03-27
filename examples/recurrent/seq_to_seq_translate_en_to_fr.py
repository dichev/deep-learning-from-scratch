import matplotlib.pyplot as plt
import torch
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import trange
import re

from models.recurrent_networks import Seq2Seq, Encoder, Decoder
from lib.optimizers import Adam
from lib.functions.losses import cross_entropy, accuracy
from lib.functions.metrics import BLEU
from lib.regularizers import grad_clip_norm_
from lib.training import batches
from preprocessing.text import clean_text, TextVocabulary

# training settings
EPOCHS = 100
BATCH_SIZE = 1024
LEARN_RATE = 0.02
DEVICE = 'cuda'

# hyperparams
ENCODER_EMBED_SIZE = 256
ENCODER_HIDDEN_SIZE = 256
DECODER_EMBED_SIZE = 256
DECODER_HIDDEN_SIZE = 256
IN_SEQ_LEN = 15
OUT_SEQ_LEN = 15

# Prepare text data
download_and_extract_archive('https://www.manythings.org/anki/fra-eng.zip', './data/text')
print('Data preprocessing..')
with open('./data/text/fra.txt', 'r', encoding='utf-8') as f: # tab-separated file
    docs_en, docs_fr, _ = zip(*[line.strip().split('\t') for line in f])

# Tokenize and index
text_tokenized_en = [clean_text(line, 'en').split() + ['<EOS>'] for line in docs_en]
text_tokenized_fr = [clean_text(line, 'fr').split() + ['<EOS>'] for line in docs_fr]

vocab_en = TextVocabulary(text_tokenized_en, min_freq=2, special_tokens=('<SOS>',))  #, special_tokens=('<SOS>', '<EOS>'))
vocab_fr = TextVocabulary(text_tokenized_fr, min_freq=2, special_tokens=('<SOS>',))  #, special_tokens=('<SOS>', '<EOS>'))
PAD_IDX = vocab_fr.to_idx['<PAD>']; assert vocab_en.to_idx['<PAD>'] == vocab_fr.to_idx['<PAD>']

text_encoded_en = torch.tensor(vocab_en.encode_batch(text_tokenized_en, seq_length=IN_SEQ_LEN), dtype=torch.long)
text_encoded_fr = torch.tensor(vocab_fr.encode_batch(text_tokenized_fr, seq_length=OUT_SEQ_LEN), dtype=torch.long)
N = len(text_encoded_en)

def diagnostics():
    print(vocab_en)
    print(vocab_fr)

    len_en = [len(seq) for seq in text_tokenized_en]
    len_fr = [len(seq) for seq in text_tokenized_fr]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.hist(len_en, bins=50, label='english', alpha=.5, color='blue')
    ax2.hist(len_fr, bins=50, label='french', alpha=.5, color='orange')
    ax1.axvline(IN_SEQ_LEN, label='max in sequence length', color='blue')
    ax2.axvline(OUT_SEQ_LEN, label='max out sequence length', color='orange')
    ax1.legend()
    ax2.legend()
    plt.suptitle('Length of sequences')
    plt.tight_layout()
    plt.show()

    vocab_en.print_human(text_encoded_en[:5].numpy())
    vocab_fr.print_human(text_encoded_fr[:5].numpy())
diagnostics()



# Model
model = Seq2Seq(
    encoder=Encoder(vocab_en.size, ENCODER_EMBED_SIZE, ENCODER_HIDDEN_SIZE, cell='lstm', n_layers=1, direction='backward', padding_idx=PAD_IDX),
    decoder=Decoder(vocab_fr.size, DECODER_EMBED_SIZE, DECODER_HIDDEN_SIZE, cell='lstm', n_layers=1, direction='forward', padding_idx=PAD_IDX),
    sos_token=vocab_fr.to_idx['<SOS>'],
)

model.to(DEVICE)
model.summary()
optimizer = Adam(model.parameters(), lr=LEARN_RATE)


def train(model, loader):
    total_loss = total_acc = 0
    pbar = trange(N, desc=f'Epoch (batch_size={BATCH_SIZE})')
    for i, (X, Y, batch_frac) in enumerate(loader):
        optimizer.zero_grad()
        Z = model.forward(X, out_steps=OUT_SEQ_LEN, targets=Y)
        cost = cross_entropy(Z, Y, logits=True, ignore_idx=PAD_IDX)
        cost.backward()
        grad_clip_norm_(model.parameters(), 1.)
        optimizer.step()

        loss, acc = cost.item(), accuracy(Z.argmax(dim=-1), Y, ignore_idx=PAD_IDX)
        total_loss += loss * batch_frac
        total_acc += acc * batch_frac

        pbar.update(BATCH_SIZE)
        pbar.set_postfix_str(f'loss={loss:.3f}, acc={acc:.3f}')

    return total_loss, total_acc, pbar


# Training
for epoch in range(EPOCHS):
    loader = batches(X=text_encoded_en, y=text_encoded_fr, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE)  # todo: migrate to DataLoader
    loss, acc, pbar = train(model, loader)
    pbar.desc = f'Epoch {epoch+1}/{EPOCHS}'; pbar.set_postfix_str(f'{loss=:.3f}, {acc=:.3f}'); pbar.close()

    # print some translations
    with torch.no_grad():
        n, offset = 5, N//2
        X, Y = text_encoded_en[offset-n:offset], text_encoded_fr[offset-n:offset]
        Z = model.forward(X.to(DEVICE), out_steps=OUT_SEQ_LEN, targets=None)
        Y_hat = Z.argmax(dim=-1).cpu()

        for x, y, y_hat in zip(X.numpy(), Y.numpy(), Y_hat.numpy()):
            source, target, predicted = vocab_en.decode(x), vocab_fr.decode(y), vocab_fr.decode(y_hat)
            source, target, predicted = [re.sub(r'<EOS>.*', '', seq) for seq in (source, target, predicted)]  # ignore everything after <EOS>
            print(f'BLEU(3): {BLEU(target.split(), predicted.split(), max_n=3):.3f} | {source} ({target}) -> {predicted}')
