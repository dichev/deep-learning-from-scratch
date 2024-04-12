import torch
from lib.layers import Module, Linear, Embedding, MultiHeadAttention, LayerNorm, Dropout, ReLU, Sequential, ModuleList, PositionalEncoding
from models.recurrent_networks import Seq2Seq
from collections import namedtuple


# todo: shared embeddings
# todo: return all the attn weighs
# todo: match param size to paper


Context = namedtuple('Context', ['memory', 'memory_pad_mask', 'cached_targets'])


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

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0, attn_heads=8, dropout=0.1, max_seq_len=1000):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.pos_emb = PositionalEncoding(embed_size, max_seq_len, dropout, mixed=True)
        self.layers = ModuleList([
            TransformerEncoderLayer(embed_size, hidden_size, attn_heads, dropout) for _ in range(n_layers)
        ])
        self.padding_idx = padding_idx
        self.vocab_size, self.embed_size, self.hidden_size = vocab_size, embed_size, hidden_size

    def forward(self, x):
        B, T = x.shape
        pad_mask = (x == self.padding_idx)

        x = self.emb.forward(x)
        x = self.pos_emb.forward(x)
        for layer in self.layers:
            x = layer.forward(x, pad_mask)

        return x, Context(memory=x, memory_pad_mask=pad_mask, cached_targets=None)



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

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0, attn_heads=8, dropout=0.1, max_seq_len=1000):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.pos_emb = PositionalEncoding(embed_size, max_seq_len, dropout, mixed=True)
        self.layers = ModuleList([
            TransformerDecoderLayer(embed_size, hidden_size, attn_heads, dropout) for _ in range(n_layers)
        ])
        self.out = Linear(embed_size, vocab_size)

        self.n_layers = n_layers
        self.padding_idx = padding_idx
        self.vocab_size, self.embed_size, self.hidden_size = vocab_size, embed_size, hidden_size


    def forward(self, tgt, context: Context):
        assert context.cached_targets is None, f'During training are expected no cached targets, use .predict() for inference'
        B, T = tgt.shape

        # Attention masks
        pad_mask = (tgt == self.padding_idx)                                          # restrict attention to padded targets
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(tgt.device)  # restrict to attend only to the previous tokens to avoid information leakage and preserve the autoregression
        attn_mask = pad_mask.unsqueeze(1) | causal_mask

        # Decode each layer
        tgt = self.emb.forward(tgt)      # (B, T, emb)
        tgt = self.pos_emb.forward(tgt)  # (B, T, emb)
        for i, layer in enumerate(self.layers):
            tgt = layer.forward(tgt, attn_mask, context.memory, context.memory_pad_mask)

        # Finally transform to logits of size (B, T, vocab_size)
        y = self.out.forward(tgt)

        return y, Context(context.memory, context.memory_pad_mask, cached_targets=None)

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

        # Finally transform to logits of size (B, T, vocab_size)
        y = self.out.forward(x)

        return y, Context(context.memory, context.memory_pad_mask, cached_targets=cache)


class Transformer(Seq2Seq):
    """
    Paper: Attention Is All You Need
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, sos_token, eos_token):
        super().__init__(encoder, decoder, sos_token, eos_token)

