import torch
from lib.layers import Module, Linear, Embedding, MultiHeadAttention, LayerNorm, Dropout, ReLU, Sequential, ModuleList
from utils.other import paddings_mask
from collections import namedtuple


# todo: positional encodings
#       In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
# todo: shared embeddings
# todo: return all the attn weighs


Context = namedtuple('Context', ['memory', 'memory_pad_mask', 'cached_targets'])


class Transformer(Module):
    """
    Attention Is All You Need
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self):
        pass

    def forward(self, x):
        pass


class TransformerEncoderLayer(Module):

    def __init__(self, input_size, hidden_size, attn_heads=1, dropout=0.):
        self.attn = MultiHeadAttention(input_size, attn_heads, scaled=True, dropout=dropout)
        self.norm1 = LayerNorm(input_size)
        self.ff = Sequential(  # Position-wise (per token)
            Linear(input_size, hidden_size), ReLU(),
            Linear(hidden_size, input_size),
            Dropout(dropout)
        )
        self.norm2 = LayerNorm(input_size)

        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size

    def forward(self, x, pad_mask):
        B, T, E = x.shape

        v, attn_weights = self.attn.forward(query=x, key=x, value=x, attn_mask=pad_mask)
        x = self.norm1.forward(x + v)
        x = self.norm2.forward(x + self.ff.forward(x))
        return x


class TransformerEncoder(Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.layers = ModuleList([
            TransformerEncoderLayer(input_size=embed_size, hidden_size=hidden_size, attn_heads=8, dropout=0.1) for _ in range(n_layers)
        ])
        self.padding_idx = padding_idx
        self.vocab_size, self.embed_size, self.hidden_size = vocab_size, embed_size, hidden_size

    def forward(self, x):
        B, T = x.shape
        pad_mask = (x == self.padding_idx)

        x = self.emb.forward(x)
        for layer in self.layers:
            x = layer.forward(x, pad_mask)

        return Context(memory=x, memory_pad_mask=pad_mask, cached_targets=None)



class TransformerDecoderLayer(Module):

    def __init__(self, input_size, hidden_size, attn_heads=1, dropout=0.):
        self.attn = MultiHeadAttention(input_size, attn_heads, scaled=True, dropout=dropout)
        self.norm1 = LayerNorm(input_size)

        self.cross_attn = MultiHeadAttention(input_size, attn_heads, scaled=True, dropout=dropout)
        self.norm2 = LayerNorm(input_size)

        self.ff = Sequential(  # Position-wise (per token)
            Linear(input_size, hidden_size), ReLU(),
            Linear(hidden_size, input_size),
            Dropout(dropout)
        )
        self.norm3 = LayerNorm(input_size)

        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size

    def forward(self, tgt, attn_mask, memory, memory_pad_mask=None, tgt_cached=None):
        B, T, E = tgt.shape

        # Self-attention
        if tgt_cached is None:
            tgt_cached = tgt  # the cached targets are used only during autoregressive inference
        v, attn_weights = self.attn.forward(query=tgt, key=tgt_cached, value=tgt_cached, attn_mask=attn_mask)
        tgt = self.norm1.forward(tgt + v)

        # Cross-attention
        v, attn_weights_mem = self.cross_attn.forward(query=tgt, key=memory, value=memory, attn_mask=memory_pad_mask)
        tgt = self.norm2.forward(tgt + v)

        # Feed forward
        tgt = self.norm3.forward(tgt + self.ff.forward(tgt))

        return tgt



class TransformerDecoder(Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.layers = ModuleList([
            TransformerDecoderLayer(input_size=embed_size, hidden_size=hidden_size, attn_heads=8, dropout=0.1) for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.padding_idx = padding_idx
        self.vocab_size, self.embed_size, self.hidden_size = vocab_size, embed_size, hidden_size


    def forward(self, tgt, context: Context):
        B, T = tgt.shape

        # Attention masks
        pad_mask = (tgt == self.padding_idx)                             # restrict attention to padded targets
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()    # restrict to attend only to the previous tokens to avoid information leakage and preserve the autoregression
        attn_mask = pad_mask.unsqueeze(1) | causal_mask

        # Decode each layer
        tgt = self.emb.forward(tgt)     # (B, T, emb)
        for i, layer in enumerate(self.layers):
            tgt = layer.forward(tgt, attn_mask, context.memory, context.memory_pad_mask)

        return tgt, Context(context.memory, context.memory_pad_mask, cached_targets=None)

    @torch.no_grad()
    def predict(self, x, context: Context):
        B, T = x.shape

        # Using cache of the previously generated tokens
        cache = context.cached_targets
        if cache is None:
            cache = [None] * self.n_layers

        # Decode each layer
        x = self.emb.forward(x)  # (B, T, emb)
        for i, layer in enumerate(self.layers):
            cache[i] = x if cache[i] is None else torch.cat((cache[i], x), dim=1)  # cache all the previous input tokens for each layer to avoid unnecessary computations
            x = layer.forward(x, attn_mask=None, memory=context.memory, memory_pad_mask=context.memory_pad_mask, tgt_cached=cache[i])   # no attn_mask because it is autoregressive, and don't have the future tokens

        return x, Context(context.memory, context.memory_pad_mask, cached_targets=cache)





pad_token = 0
start_token = 1
vocab_size, embed_size, hidden_size = 1000, 512, 2048
B, T = 8, 15
x_pad_mask = torch.arange(T).expand(B, T) >= torch.randint(1, T-1, (B, 1))
x = torch.randint(2, 1000, (B, T)) * ~x_pad_mask
y_shifted = torch.cat((
    torch.full((B, 1), start_token),
    x[:, :-6]  # skip last 6 tokens
), dim=1)
y_shifted_pad_mask = y_shifted == pad_token


encoder = TransformerEncoder(vocab_size, embed_size, hidden_size, padding_idx=pad_token)
decoder = TransformerDecoder(vocab_size, embed_size, hidden_size, padding_idx=pad_token)
encoder.summary()
decoder.summary()


# Training step
context = encoder.forward(x)
y_hat = decoder.forward(y_shifted, context)


# Autoregressive predictions
with torch.no_grad():
    context = encoder.forward(x)
    token = torch.full((B, 1), start_token)
    decoded_tokens = []
    for n in range(10):
        pred, context = decoder.predict(token, context)
        token = pred.argmax(-1)
        decoded_tokens.append(token)

