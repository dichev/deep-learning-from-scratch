import matplotlib.pyplot as plt
import torch
import math
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import trange

from lib.functions.losses import cross_entropy
from lib.functions.metrics import BLEU, accuracy
from lib.regularizers import grad_clip_norm_
from lib.training import batches
from preprocessing.text import clean_text, TextVocabulary


# hyperparams
IN_SEQ_LEN = 15
OUT_SEQ_LEN = 16

# Prepare text data
print('Data loading..')
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



def train(model, optimizer, loader, batch_size):
    total_loss = total_acc = total_grad_norm = 0
    pbar = trange(N, desc=f'Epoch (batch_size={batch_size})')
    for i, (X, Y, batch_frac) in enumerate(loader):
        optimizer.zero_grad()
        Z = model.forward(X, targets=Y)
        cost = cross_entropy(Z, Y, logits=True, ignore_idx=PAD_IDX)
        cost.backward()
        grad_clip_norm_(model.parameters(), 1.)
        optimizer.step()

        loss, acc = cost.item(), accuracy(Z.argmax(dim=-1), Y, ignore_idx=PAD_IDX)
        total_loss += loss * batch_frac
        total_acc += acc * batch_frac
        total_grad_norm += model.grad_norm() * batch_frac

        pbar.update(batch_size)
        pbar.set_postfix_str(f'loss={loss:.3f}, acc={acc:.3f}')

    return total_loss, total_acc, total_grad_norm, pbar


def fit(model, optimizer, epochs=100, batch_size=1024, device='cuda', title=''):
    writer = SummaryWriter(f'runs/{title or model.__class__.__name__} params={model.n_params} - {datetime.now().strftime('%b%d %H-%M-%S')}', flush_secs=2)

    for epoch in range(1, epochs+1):
        loader = batches(X=text_encoded_en, y=text_encoded_fr, batch_size=batch_size, shuffle=True, device=device)  # todo: migrate to DataLoader
        loss, acc, grad_norm, pbar = train(model, optimizer, loader, batch_size)
        pbar.desc = f'Epoch {epoch}/{epochs}'; pbar.set_postfix_str(f'{loss=:.3f}, {acc=:.3f}'); pbar.close()

        # Metrics
        writer.add_scalar('t/Loss', loss, epoch)
        writer.add_scalar('t/Perplexity', math.exp(loss), epoch)
        writer.add_scalar('a/Gradients Norm', grad_norm, epoch)
        writer.add_scalar('a/Weights Norm', model.weight_norm(), epoch)
        for name, param in model.parameters():
            if 'bias' not in name:
                writer.add_histogram('params/' + name.replace('.', '/'), param, epoch)
                writer.add_histogram('grad/' + name.replace('.', '/'), param.grad, epoch)  # note this is a sample from the last mini-batch


        # print some translations
        if epoch % 10 == 1:
            with torch.no_grad():
                n, offset = 5, N//2
                X, Y = text_encoded_en[offset-n:offset], text_encoded_fr[offset-n:offset]

                for i in range(n):
                    source, target = vocab_en.decode(X[i].numpy(), trim_after='<EOS>'), vocab_fr.decode(Y[i].numpy(), trim_after='<EOS>')
                    print(f'{source} (expected: {target})')
                    for beam_width in (1, 2, 3):
                        Y_hat, score = model.predict(X[i:i+1].to(device), max_steps=OUT_SEQ_LEN, beam_width=beam_width)
                        predicted = vocab_fr.decode(Y_hat.squeeze(0).numpy(), trim_after='<EOS>')
                        print(f'-> BLEU(n_grams=4): {BLEU(target.split(), predicted.split(), max_n=4):.3f} | beam=(width={beam_width}, score={score})  -> {predicted}')
                    print('------------------------------------------')

