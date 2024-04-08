import torch
from lib.layers import Module, Linear, Embedding, DotProductAttention, LayerNorm, Dropout, ReLU, Sequential, ModuleList
from lib.functions.activations import relu


# todo: learnable self-attention
# todo: multi-head attention
# todo: positional encodings
#       In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
# todo: shared embeddings


class Transformer(Module):
    """
    Attention Is All You Need
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self):
        pass

    def forward(self, x):
        pass


class TransformerLayer(Module):

    def __init__(self, input_size, hidden_size, dropout=0.):
        self.attn = DotProductAttention(scaled=True, dropout=dropout)
        self.norm1 = LayerNorm(input_size)
        self.ff = Sequential(                        # Position-wise (per token)
            Linear(input_size, hidden_size), ReLU(),
            Linear(hidden_size, input_size)
        )
        self.norm2 = LayerNorm(input_size)

        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size

    def forward(self, x, pad_mask):
        B, T, E = x.shape

        v, attn_weights = self.attn.forward(query=x, key=x, value=x, attn_mask=pad_mask.unsqueeze(1))
        x = self.norm1.forward(x + v)
        x = self.norm2.forward(x + self.ff.forward(x))

        return x


class TransformerEncoder(Module):

    def __init__(self, vocab_size, embed_size, n_layers=6, padding_idx=0):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.layers = ModuleList([
            TransformerLayer(input_size=512, hidden_size=2048, dropout=0.1) for _ in range(n_layers)
        ])
        self.padding_idx = padding_idx
        pass

    def forward(self, x):
        B, T = x.shape
        pad_mask = (x != self.padding_idx)

        x = self.emb.forward(x)
        for layer in self.layers:
            x = layer.forward(x, pad_mask)

        return x




vocab_size, embed_size = 1000, 512  # todo embed_size=64
B, T = 8, 15
x_pad_mask = torch.arange(T).expand(B, T) < torch.randint(1, T-1, (B, 1))
x = torch.randint(1, 1000, (B, T)) * x_pad_mask
encoder = TransformerEncoder(vocab_size, embed_size, padding_idx=0)
encoder.summary()
y = encoder.forward(x)
