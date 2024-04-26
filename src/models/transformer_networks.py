import torch
from lib.layers import Module, Linear, Embedding, MultiHeadAttention, LayerNorm, Dropout, ReLU, Sequential, ModuleList, PositionalEncoding
from models.recurrent_networks import Seq2Seq
from collections import namedtuple
from math import sqrt
from utils import plots



Context = namedtuple('Context', ['memory', 'memory_pad_mask', 'cached_targets'])


class TransformerEncoderLayer(Module):

    def __init__(self, input_size, hidden_size, attn_heads=1, dropout=0., norm_first=True):  # norm_first=True to avoid tuning warm-up learning rate (see https://arxiv.org/pdf/2002.04745v1.pdf)
        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(input_size)
        self.attn = MultiHeadAttention(input_size, attn_heads, scaled=True, dropout=dropout)
        self.ff = Sequential(  # Position-wise (per token)
            Linear(input_size, hidden_size), ReLU(),
            Linear(hidden_size, input_size),
            Dropout(dropout)
        )

        self.norm_first = norm_first
        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size
        self.attn_weights = None  # keep record of the last attention weights for visualization

    def forward(self, x, pad_mask):
        B, T, E = x.shape

        if self.norm_first:
            x = x + self._self_attention(self.norm1(x), attn_mask=pad_mask)
            x = x + self.norm2(self.ff(x))
        else:
            x = self.norm1(x + self._self_attention(x, attn_mask=pad_mask))
            x = self.norm2(x + self.ff(x))

        return x

    def _self_attention(self, x, attn_mask):
        v, self.attn_weights = self.attn(query=x, key=x, value=x, attn_mask=attn_mask)
        return v


class TransformerEncoder(Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0, attn_heads=8, dropout=0.1, max_seq_len=1000, norm_first=True, scale_up_embeddings=False):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.pos_emb = PositionalEncoding(embed_size, max_seq_len, dropout, mixed=True)
        self.layers = ModuleList([
            TransformerEncoderLayer(embed_size, hidden_size, attn_heads, dropout, norm_first) for _ in range(n_layers)
        ])
        if norm_first:
            self.final_norm = LayerNorm(embed_size)

        self.norm_first = norm_first
        self.padding_idx = padding_idx
        self.vocab_size, self.embed_size, self.hidden_size = vocab_size, embed_size, hidden_size
        self.emb_scale_factor = sqrt(self.embed_size) if scale_up_embeddings else 1

    def forward(self, x):
        B, T = x.shape
        pad_mask = (x == self.padding_idx)

        x = self.emb(x) * self.emb_scale_factor
        x = self.pos_emb(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, pad_mask)

        if self.norm_first:
            x = self.final_norm(x)

        return x, Context(memory=x, memory_pad_mask=pad_mask, cached_targets=None)

    def get_last_attn_weights(self):
        attn_weights = torch.stack([layer.attn_weights for layer in self.layers], dim=1)
        return attn_weights  # (b, n_layers, n_heads, t, t)




class TransformerDecoderLayer(Module):

    def __init__(self, input_size, hidden_size, attn_heads=1, dropout=0., norm_first=True):  # norm_first=True to avoid tuning warm-up learning rate (see https://arxiv.org/pdf/2002.04745v1.pdf)
        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(input_size)
        self.norm3 = LayerNorm(input_size)

        self.attn = MultiHeadAttention(input_size, attn_heads, scaled=True, dropout=dropout)
        self.cross_attn = MultiHeadAttention(input_size, attn_heads, scaled=True, dropout=dropout)
        self.ff = Sequential(  # Position-wise (per token)
            Linear(input_size, hidden_size), ReLU(),
            Linear(hidden_size, input_size),
            Dropout(dropout)
        )

        self.norm_first = norm_first
        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size
        self.attn_weights = None        # keep record of the last attention weights for visualization
        self.cross_attn_weights = None  #

    def forward(self, x, attn_mask, memory, memory_pad_mask=None, x_cached=None):
        B, T, E = x.shape

        if self.norm_first:
            x = x + self._self_attention(self.norm1(x), attn_mask, self.norm1(x_cached) if x_cached is not None else None)  # apply normalization also to the cached targets during autoregressive inference
            x = x + self._cross_attention(self.norm2(x), memory, memory_pad_mask)
            x = x + self.ff(self.norm3(x))
        else:
            x = self.norm1(x + self._self_attention(x, attn_mask, x_cached))
            x = self.norm2(x + self._cross_attention(x, memory, memory_pad_mask))
            x = self.norm3(x + self.ff(x))

        return x

    def _self_attention(self, x, attn_mask, x_cached=None):
        if x_cached is None:
            x_cached = x  # the cached targets are used only during autoregressive inference
        v, self.attn_weights = self.attn(query=x, key=x_cached, value=x_cached, attn_mask=attn_mask)
        return v

    def _cross_attention(self, x, memory, attn_mask):
        v, self.cross_attn_weights = self.cross_attn(query=x, key=memory, value=memory, attn_mask=attn_mask)
        return v



class TransformerDecoder(Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0, attn_heads=8, dropout=0.1, max_seq_len=1000, norm_first=True, tied_embeddings=False, scale_up_embeddings=False):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.pos_emb = PositionalEncoding(embed_size, max_seq_len, dropout, mixed=True)
        self.layers = ModuleList([
            TransformerDecoderLayer(embed_size, hidden_size, attn_heads, dropout, norm_first) for _ in range(n_layers)
        ])
        if norm_first:
            self.final_norm = LayerNorm(embed_size)
        if not tied_embeddings:  # note in the paper they share the embeddings also with the encoder embeddings, but that's require single (union) source/target vocabulary
            self.out = Linear(embed_size, vocab_size)

        self.norm_first = norm_first
        self.n_layers = n_layers
        self.padding_idx = padding_idx
        self.tied_embeddings = tied_embeddings
        self.vocab_size, self.embed_size, self.hidden_size = vocab_size, embed_size, hidden_size
        self.emb_scale_factor = sqrt(self.embed_size) if scale_up_embeddings else 1


    def forward(self, tgt, context: Context):
        assert context.cached_targets is None, f'During training are expected no cached targets, use .predict() for inference'
        B, T = tgt.shape

        # Attention masks
        pad_mask = (tgt == self.padding_idx)                                          # restrict attention to padded targets
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(tgt.device)  # restrict to attend only to the previous tokens to avoid information leakage and preserve the autoregression
        attn_mask = pad_mask.unsqueeze(1) | causal_mask

        # Decode each layer
        tgt = self.emb(tgt) * self.emb_scale_factor    # (B, T, emb)
        tgt = self.pos_emb(tgt)                        # (B, T, emb)
        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, attn_mask, context.memory, context.memory_pad_mask)

        if self.norm_first:
            tgt = self.final_norm(tgt)

        # Finally transform to logits of size (B, T, vocab_size)
        if self.tied_embeddings:
            y = self.emb.backward(tgt)
        else:
            y = self.out(tgt)

        return y, Context(context.memory, context.memory_pad_mask, cached_targets=None)

    @torch.no_grad()
    def predict(self, x, context: Context):
        B, T = x.shape

        # Using cache of the previously generated tokens
        cache = context.cached_targets
        if cache is None:
            cache = [None] * self.n_layers

        # Decode each layer
        x = self.emb(x) * self.emb_scale_factor  # (B, T, emb)
        for i, layer in enumerate(self.layers):
            cache[i] = x if cache[i] is None else torch.cat((cache[i], x), dim=1)  # cache all the previous input tokens for each layer to avoid unnecessary computations
            x = layer(x, attn_mask=None, memory=context.memory, memory_pad_mask=context.memory_pad_mask, x_cached=cache[i])   # no attn_mask because it is autoregressive, and don't have the future tokens

        if self.norm_first:
            x = self.final_norm(x)

        # Finally transform to logits of size (B, T, vocab_size)
        if self.tied_embeddings:
            y = self.emb.backward(x)
        else:
            y = self.out(x)

        return y, Context(context.memory, context.memory_pad_mask, cached_targets=cache)

    def get_last_attn_weights(self):
        self_attn_weights = torch.stack([layer.attn_weights for layer in self.layers], dim=1)
        cross_attn_weights = torch.stack([layer.cross_attn_weights for layer in self.layers], dim=1)
        return self_attn_weights, cross_attn_weights  # (b, n_layers, n_heads, t', t'), (b, n_layers, n_heads, t', t)


class Transformer(Seq2Seq):
    """
    Paper: Attention Is All You Need
    https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, sos_token, eos_token):
        super().__init__(encoder, decoder, sos_token, eos_token)


    def visualize_attn_weights(self, src_labels, tgt_labels, subtitle=''):
        encoder_attn = self.encoder.get_last_attn_weights()
        decoder_self_attn, decoder_cross_attn = self.decoder.get_last_attn_weights()
        plots.attention_heads(encoder_attn[0].detach().cpu(), src_labels, src_labels, title=f'Encoder Self-Attention {subtitle}')
        plots.attention_heads(decoder_self_attn[0].detach().cpu(), tgt_labels, tgt_labels, title=f'Decoder Self-Attention {subtitle}')
        plots.attention_heads(decoder_cross_attn[0].detach().cpu(), tgt_labels, src_labels, title=f'Decoder Cross-Attention {subtitle}')


