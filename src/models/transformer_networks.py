import torch
import einops as ein
from lib.layers import Module, Linear, Embedding, MultiHeadAttention, LayerNorm, Dropout, ReLU, GELU, Sequential, ModuleList, PositionalEncoding
from lib.functions.activations import softmax
from models.recurrent_networks import Seq2Seq
from collections import namedtuple
from math import sqrt
from utils import plots



Context = namedtuple('Context', ['memory', 'memory_attn_pad_mask', 'cached_targets'])


class TransformerEncoderLayer(Module):

    def __init__(self, input_size, hidden_size, attn_heads=1, dropout=0., norm_first=True, gelu_activation=True):  # norm_first=True to avoid tuning warm-up learning rate (see https://arxiv.org/pdf/2002.04745v1.pdf)
        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(input_size)
        self.attn = MultiHeadAttention(input_size, attn_heads, dropout=dropout)
        self.ff = Sequential(  # Position-wise (per token)
            Linear(input_size, hidden_size),
            GELU() if gelu_activation else ReLU(),
            Linear(hidden_size, input_size),
            Dropout(dropout)
        )

        self.norm_first = norm_first
        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size
        self.attn_weights = None  # keep record of the last attention weights for visualization

    def forward(self, x, attn_mask):
        B, T, E = x.shape

        if self.norm_first:
            x = x + self._self_attention(self.norm1(x), attn_mask)
            x = x + self.norm2(self.ff(x))
        else:
            x = self.norm1(x + self._self_attention(x, attn_mask))
            x = self.norm2(x + self.ff(x))

        return x

    def _self_attention(self, x, attn_mask):
        v, self.attn_weights = self.attn(query=x, key=x, value=x, attn_mask=attn_mask)
        return v


class TransformerEncoder(Module):

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0, attn_heads=8, dropout=0.1, max_seq_len=1000, norm_first=True, gelu_activation=True, scale_up_embeddings=False):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.pos_emb = PositionalEncoding(embed_size, max_seq_len, dropout, mixed=True)
        self.layers = ModuleList([
            TransformerEncoderLayer(embed_size, hidden_size, attn_heads, dropout, norm_first, gelu_activation) for _ in range(n_layers)
        ])
        if norm_first:
            self.final_norm = LayerNorm(embed_size)

        self.norm_first = norm_first
        self.padding_idx = padding_idx
        self.vocab_size, self.embed_size, self.hidden_size = vocab_size, embed_size, hidden_size
        self.emb_scale_factor = sqrt(self.embed_size) if scale_up_embeddings else 1
        self.n_heads = attn_heads

    def forward(self, x):
        B, T = x.shape
        pad_mask = (x == self.padding_idx)  # (B, T)
        attn_mask = ein.repeat(pad_mask, "b t -> (b h) 1 t", h=self.n_heads)  # expected shape of attn_mask is (b*h, t', t), so repeat pad_mask along head sub-dimensions

        x = self.emb(x) * self.emb_scale_factor
        x = self.pos_emb(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask)

        if self.norm_first:
            x = self.final_norm(x)

        return x, Context(memory=x, memory_attn_pad_mask=attn_mask, cached_targets=None)

    def get_last_attn_weights(self):
        attn_weights = torch.stack([layer.attn_weights for layer in self.layers], dim=1)
        return attn_weights  # (b, n_layers, n_heads, t, t)




class TransformerDecoderLayer(Module):

    def __init__(self, input_size, hidden_size, attn_heads=1, dropout=0., norm_first=True, gelu_activation=True):  # norm_first=True to avoid tuning warm-up learning rate (see https://arxiv.org/pdf/2002.04745v1.pdf)
        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(input_size)
        self.norm3 = LayerNorm(input_size)

        self.attn = MultiHeadAttention(input_size, attn_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(input_size, attn_heads, dropout=dropout)
        self.ff = Sequential(  # Position-wise (per token)
            Linear(input_size, hidden_size),
            GELU() if gelu_activation else ReLU(),
            Linear(hidden_size, input_size),
            Dropout(dropout)
        )

        self.norm_first = norm_first
        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size
        self.attn_weights = None        # keep record of the last attention weights for visualization
        self.cross_attn_weights = None  #

    def forward(self, x, attn_mask, memory, memory_attn_pad_mask=None, x_cached=None):
        B, T, E = x.shape

        if self.norm_first:
            x = x + self._self_attention(self.norm1(x), attn_mask, self.norm1(x_cached) if x_cached is not None else None)  # apply normalization also to the cached targets during autoregressive inference
            x = x + self._cross_attention(self.norm2(x), memory, memory_attn_pad_mask)
            x = x + self.ff(self.norm3(x))
        else:
            x = self.norm1(x + self._self_attention(x, attn_mask, x_cached))
            x = self.norm2(x + self._cross_attention(x, memory, memory_attn_pad_mask))
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

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=6, padding_idx=0, attn_heads=8, dropout=0.1, max_seq_len=1000, norm_first=True, tied_embeddings=False, gelu_activation=True, scale_up_embeddings=False):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.pos_emb = PositionalEncoding(embed_size, max_seq_len, dropout, mixed=True)
        self.layers = ModuleList([
            TransformerDecoderLayer(embed_size, hidden_size, attn_heads, dropout, norm_first, gelu_activation) for _ in range(n_layers)
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
        self.n_heads = attn_heads


    def forward(self, tgt, context: Context):
        assert context.cached_targets is None, f'During training are expected no cached targets, use .predict() for inference'
        B, T = tgt.shape

        # Attention masks
        pad_mask = (tgt == self.padding_idx)                                          # restrict attention to padded targets
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(tgt.device)  # restrict to attend only to the previous tokens to avoid information leakage and preserve the autoregression
        attn_mask = causal_mask | ein.repeat(pad_mask, "b t -> (b h) 1 t", h=self.n_heads)

        # Decode each layer
        tgt = self.emb(tgt) * self.emb_scale_factor    # (B, T, emb)
        tgt = self.pos_emb(tgt)                        # (B, T, emb)
        for i, layer in enumerate(self.layers):
            tgt = layer(tgt, attn_mask, context.memory, context.memory_attn_pad_mask)

        if self.norm_first:
            tgt = self.final_norm(tgt)

        # Finally transform to logits of size (B, T, vocab_size)
        if self.tied_embeddings:
            y = self.emb.backward(tgt)
        else:
            y = self.out(tgt)

        return y, Context(context.memory, context.memory_attn_pad_mask, cached_targets=None)

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
            x = layer(x, attn_mask=None, memory=context.memory, memory_attn_pad_mask=context.memory_attn_pad_mask, x_cached=cache[i])   # no attn_mask because it is autoregressive, and don't have the future tokens

        if self.norm_first:
            x = self.final_norm(x)

        # Finally transform to logits of size (B, T, vocab_size)
        if self.tied_embeddings:
            y = self.emb.backward(x)
        else:
            y = self.out(x)

        return y, Context(context.memory, context.memory_attn_pad_mask, cached_targets=cache)

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


    def visualize_attn_weights(self, src_labels, tgt_labels, subtitle='', batch_idx=0):
        encoder_attn = self.encoder.get_last_attn_weights()
        decoder_self_attn, decoder_cross_attn = self.decoder.get_last_attn_weights()
        plots.attention_heads(encoder_attn[batch_idx].detach().cpu(), src_labels, src_labels, title=f'Encoder Self-Attention {subtitle}')
        plots.attention_heads(decoder_self_attn[batch_idx].detach().cpu(), tgt_labels, tgt_labels, title=f'Decoder Self-Attention {subtitle}')
        plots.attention_heads(decoder_cross_attn[batch_idx].detach().cpu(), tgt_labels, src_labels, title=f'Decoder Cross-Attention {subtitle}')



class GPT2_Block(Module):  # Same as TransformerDecoderLayer but without cross-attention

    def __init__(self, input_size, hidden_size, attn_heads, max_seq_len, dropout=0.):  # norm_first=True to avoid tuning warm-up learning rate (see https://arxiv.org/pdf/2002.04745v1.pdf)
        self.norm1 = LayerNorm(input_size)
        self.attn = MultiHeadAttention(input_size, attn_heads, dropout=dropout)
        self.norm2 = LayerNorm(input_size)
        self.ff = Sequential(  # Position-wise (per token)
            Linear(input_size, hidden_size),
            GELU(),
            Linear(hidden_size, input_size),
            Dropout(dropout)
        )
        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, input_size
        self.attn_weights = None        # keep record of the last attention weights for visualization
        self.causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()

    def forward(self, x):
        x = x + self._self_attention(self.norm1(x))  # apply normalization also to the cached targets during autoregressive inference
        x = x + self.ff(self.norm2(x))
        return x

    def _self_attention(self, x):
        B, T, E = x.shape

        # Attention mask
        attn_mask = self.causal_mask[:T, :T].to(x.device)  # restrict to attend only to the previous tokens to avoid information leakage and preserve the autoregression
        v, self.attn_weights = self.attn(query=x, key=x, value=x, attn_mask=attn_mask)
        return v


class GPT2(Module):
    """
    Paper: Language Models are Unsupervised Multitask Learners
    https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    def __init__(self, vocab_size=50_257, context_size=1024, embed_size=768, hidden_size=768*4, n_layers=12, attn_heads=12, dropout=0.1):
        self.emb = Embedding(vocab_size, embed_size)
        self.pos_emb = Embedding(context_size, embed_size)
        self.dropout = Dropout(dropout)
        self.transformers = ModuleList(
            GPT2_Block(embed_size, hidden_size, attn_heads, max_seq_len=context_size, dropout=dropout) for _ in range(n_layers)
        )
        self.final_norm = LayerNorm(embed_size)

        self.n_layers = n_layers
        self.vocab_size, self.context_size, self.embed_size, self.hidden_size = vocab_size, context_size, embed_size, hidden_size
        self.reset_parameters()  # Custom parameter initializations


    @torch.no_grad()
    def reset_parameters(self):
        # GPT-1: "Since layer norm is used extensively throughout the model, a simple weight initialization of N (0, 0.02) was sufficient."
        self.emb.weight.data.normal_(std=0.02)
        self.pos_emb.weight.data.normal_(std=0.01)
        [layer_norm.reset_parameters() for name, layer_norm in self.modules() if isinstance(layer_norm, LayerNorm)]
        # GPT-2: "We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers"
        for name, param in self.transformers.parameters():
            if 'weight' in name: param.data.normal_(std=0.02 / sqrt(2 * self.n_layers))  # check: just the projections, or also the ff
            elif 'bias' in name: param.data.zero_()

    def forward(self, x):
        B, T = x.shape
        assert T <= self.context_size, f'the input sequence {T} exceeds the context size {self.context_size}'
        positions = torch.arange(T, device=x.device)

        # Decode each token
        x = self.emb(x) + self.pos_emb(positions)  # (B, T, emb) + (T, emb) -> (B, T, emb)
        x = self.dropout(x)
        for i, layer in enumerate(self.transformers):
            x = layer(x)
        x = self.final_norm(x)

        # Output token logits
        z = self.emb.backward(x)  # reuse the embedding matrix weights during pretraining
        return z

    def get_last_attn_weights(self):
        return torch.stack([layer.attn_weights for layer in self.transformers], dim=1)   # (b, n_layers, n_heads, t, t)

    def visualize_attn_weights(self, batch_idx=0, subtitle=''):
        attn_weights = self.get_last_attn_weights().detach().cpu()
        plots.attention_heads_fast(attn_weights[batch_idx], title=f'Self-Attention {subtitle}')

    @torch.no_grad()
    def generate(self, x, max_tokens=10):  # note: caching of previous token hidden states is not implemented to keep the training code cleaner (see TransformerDecoder for such an implementation)
        for i in range(max_tokens):
            x = x[:, -self.context_size:]              # limit the input to not exceed the context size (because the fixed size of positional emb)
            z = self.forward(x)                        # (B, T, E)
            p = softmax(z[:, -1, :], dim=-1)           # just the last token
            y = torch.multinomial(p, num_samples=1)    # (B, 1)
            x = torch.cat((x, y), dim=1)       # (B, T+1)
            yield y

