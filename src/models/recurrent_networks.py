import torch
from lib.layers import Module, Linear, RNN, LayerNorm, ModuleList
from lib.functions import init
from lib.functions.activations import softmax
from preprocessing.integer import one_hot


class RNN_factory(Module):

    def __init__(self, input_size, hidden_size, cell='rnn', n_layers=1, direction='forward', layer_norm=False):
        assert direction in ('forward', 'backward', 'bidirectional'), "direction must be one of ('forward', 'backward', 'bidirectional')"
        self.n_layers = n_layers
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

    def init_state(self):
        if self.direction != 'bidirectional':
            return [(None, None)] * self.n_layers                  # states[i] = [(h, C)]
        else:
            return [[(None, None), (None, None)]] * self.n_layers  # states[i] = [(h_f, C_f), (h_b, C_b)]

    def forward(self, x, states=None):
        if len(x.shape) == 2:  # when x is indices
            x = one_hot(x, self.input_size)

        if states is None:
            states = self.init_state()

        for i in range(self.n_layers):
            if self.direction != 'bidirectional':
                z, states[i] = self.rnn_layers[i].forward(x, states[i])
            else:
                z_f, states[i][0] = self.rnn_layers_f[i].forward(x, states[i][0])
                z_b, states[i][1] = self.rnn_layers_b[i].forward(x, states[i][1])
                z = torch.concat((z_f, z_b), dim=-1)

            x = z  # feed next layers with the output of the previous layer

        return z, states


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