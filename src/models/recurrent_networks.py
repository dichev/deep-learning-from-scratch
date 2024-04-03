import torch
from collections import namedtuple
from lib.layers import Module, Linear, RNN, ModuleList, Embedding
from lib.functions import init
from lib.functions.activations import softmax
from preprocessing.integer import one_hot
from utils.search import beam_search, greedy_search

State = namedtuple('State', ['hidden', 'cell'])

class RNN_factory(Module):

    def __init__(self, input_size, hidden_size, cell='rnn', n_layers=1, direction='forward', layer_norm=False):
        assert direction in ('forward', 'backward', 'bidirectional'), "direction must be one of ('forward', 'backward', 'bidirectional')"
        self.n_layers = n_layers
        self.is_doubled = 2 if direction == 'bidirectional' else 1
        self.input_size = input_size
        self.direction = direction
        self.cell_type = cell

        if direction != 'bidirectional':
            self.hidden_size = hidden_size
            self.rnn_layers = ModuleList([RNN(input_size if i == 0 else self.hidden_size, self.hidden_size, cell=cell, backward=(direction == 'backward'), layer_norm=layer_norm) for i in range(n_layers)])
        else:
            self.hidden_size = hidden_size * 2
            self.rnn_layers_f = ModuleList([RNN(input_size if i == 0 else self.hidden_size, self.hidden_size//2, cell=cell, backward=False, layer_norm=layer_norm) for i in range(n_layers)])
            self.rnn_layers_b = ModuleList([RNN(input_size if i == 0 else self.hidden_size, self.hidden_size//2, cell=cell, backward=True, layer_norm=layer_norm) for i in range(n_layers)])

    def empty_state(self):
        return [None] * self.n_layers * self.is_doubled

    def forward(self, x, states=None):
        if len(x.shape) == 2:  # when x is indices
            x = one_hot(x, self.input_size)
        B, T, _ = x.shape

        h0 = self.empty_state() if (states is None or states[0] is None) else states[0]
        c0 = self.empty_state() if (states is None or states[1] is None) else states[1]
        h = torch.zeros(self.n_layers * self.is_doubled, B, self.hidden_size // self.is_doubled, device=x.device)
        c = torch.zeros(self.n_layers * self.is_doubled, B, self.hidden_size // self.is_doubled, device=x.device) if self.cell_type == 'lstm' else self.empty_state()
        for i in range(self.n_layers):
            if self.direction != 'bidirectional':
                z, (h[i],  c[i]) = self.rnn_layers[i].forward(x, (h0[i], c0[i]))
            else:
                f, b = i*2, i*2+1
                z_f, (h[f], c[f]) = self.rnn_layers_f[i].forward(x, (h0[f], c0[f]))
                z_b, (h[b], c[b]) = self.rnn_layers_b[i].forward(x, (h0[b], c0[b]))
                z = torch.concat((z_f, z_b), dim=-1)

            x = z  # feed next layers with the output of the previous layer

        return z, State(h, c)


class SimpleRNN(RNN_factory):
    def __init__(self, input_size, hidden_size, n_layers=1, direction='forward', layer_norm=False):
        super().__init__(input_size, hidden_size, cell='rnn', n_layers=n_layers, direction=direction, layer_norm=layer_norm)


class LSTM(RNN_factory):
    """
    Paper: Generating Sequences With Recurrent Neural Networks
    https://arxiv.org/pdf/1308.0850.pdf
    """
    def __init__(self, input_size, hidden_size, n_layers=1, direction='forward', layer_norm=False):
        super().__init__(input_size, hidden_size, cell='lstm', n_layers=n_layers, direction=direction, layer_norm=layer_norm)


class GRU(RNN_factory):
    def __init__(self, input_size, hidden_size, n_layers=1, direction='forward', layer_norm=False):
        super().__init__(input_size, hidden_size, cell='gru', n_layers=n_layers, direction=direction, layer_norm=layer_norm)


class LangModel(Module):
    def __init__(self, rnn: RNN_factory):
        self.vocab_size = rnn.input_size
        self.rnn = rnn
        self.head = Linear(self.rnn.hidden_size, self.vocab_size, weights_init=init.xavier_normal_)

    def forward(self, x, states=None):
        h, states = self.rnn.forward(x, states)
        z = self.head.forward(h)
        return z, states

    @torch.no_grad()
    def sample(self, n_samples=1, temperature=1., seed_seq=None):
        if seed_seq is None:
            x = torch.randint(0, self.vocab_size, size=(1,), device=self.head.weight.device)
        else:
            x = torch.tensor(seed_seq, device=self.head.weight.device)

        states = None
        seq = []
        for n in range(n_samples):
            z, states = self.forward(x.view(1, len(x)), states)
            p = softmax(z[0][-1] / temperature)
            token = torch.multinomial(p, num_samples=1)  # sample single token
            seq.append(token.item())
            x = token

        return seq


class EchoStateNetwork(Module):

    def __init__(self, input_size, hidden_size, output_size, spectral_radius=2., sparsity=.90):
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.rnn = RNN(input_size, hidden_size)
        self.out = Linear(hidden_size, output_size, weights_init=init.xavier_normal_)

        # make all input-hidden and hidden-hidden parameters fixed, to be used as a reservoir
        for param in self.rnn.parameters(named=False):
            param.requires_grad = False

        # tune the spectral radius of the hidden-hidden weights
        Whh = self.rnn.cell.weight[input_size : input_size + hidden_size]
        Whh *= torch.rand_like(Whh) > sparsity          # set 90% sparse connections:
        Whh /= torch.linalg.eigvals(Whh).abs().max()    # set the spectral radius of the hidden-hidden weights to 1
        Whh *= spectral_radius                          # scale up the spectral radius to 2, because the tanh saturation

    def forward(self, x, h=(None, None)):
        if len(x.shape) == 2:  # when x is indices
            x = one_hot(x, self.input_size)

        z, h = self.rnn.forward(x, h)
        y = self.out.forward(z)
        return y, h

    def to(self, device):
        super().to(device)
        # those parameters aren't moved because aren't trainable (required_grad=False)
        self.rnn.cell.weight.data = self.rnn.cell.weight.data.to(device)
        self.rnn.cell.bias.data = self.rnn.cell.bias.data.to(device)


class Encoder(Module):
    def __init__(self, vocab_size, embed_size, hidden_size, cell='lstm', n_layers=4, direction='backward', layer_norm=False, padding_idx=None):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        # note the input direction is in backward: [paper] "we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM’s performance markedly"
        self.rnn = RNN_factory(embed_size, hidden_size, cell, n_layers, direction, layer_norm)

    def forward(self, x):
        x = self.emb.forward(x)                     # (B, T) -> (B, T, embed_size)
        output, context = self.rnn.forward(x)       # (B, T, embed_size)) -> (B, T, hidden_size), [(h, C)]
        return output, context


class Decoder(Module):
    def __init__(self, vocab_size, embed_size, hidden_size, cell='lstm', n_layers=4, direction='forward', layer_norm=False, padding_idx=None):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.rnn = RNN_factory(embed_size, hidden_size, cell, n_layers, direction, layer_norm)
        self.out = Linear(hidden_size, vocab_size, weights_init=init.xavier_normal_)

    def forward(self, x, context):
        x = self.emb.forward(x)                               # (B, T) -> (B, T, embed_size)
        output, states = self.rnn.forward(x, states=context)  # (B, T, embed_size)) -> (B, T, hidden_size), [(h, C)]
        y = self.out.forward(output)
        return y, states


class Seq2Seq(Module):
    """
    Paper: Sequence to Sequence Learning with Neural Networks
    https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf
    """

    def __init__(self, encoder, decoder, sos_token, eos_token):
        self.encoder = encoder
        self.decoder = decoder
        self.sos_token = sos_token
        self.eos_token = eos_token

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape

        # Encode all tokens into single fixed-size context
        _, context = self.encoder.forward(x)

        # Decode the context using Teacher forcing
        start_token = torch.full((batch_size, 1), self.sos_token).long().to(x.device)
        targets = torch.cat([start_token, targets[:, :-1]], dim=1)
        outputs, states = self.decoder.forward(targets, context)

        return outputs  # (batch_size, out_seq_len, out_vocab_size)

    @torch.no_grad()
    def predict(self, x, max_steps, beam_width=1):
        batch_size, seq_len = x.shape
        assert batch_size == 1, f'batched predictions aren\'t supported'

        # Encode all tokens into single fixed-size context
        _, context = self.encoder.forward(x)

        # Decode the context into sequence of tokens one by one
        if beam_width == 1:  # greedy search
            seq, score = greedy_search(self.decoder, context, self.sos_token, self.eos_token, max_steps), None
        else:
            seq, score = beam_search(self.decoder, context, self.sos_token, self.eos_token, max_steps, k=beam_width)

        return torch.tensor(seq).view(1, -1), score


