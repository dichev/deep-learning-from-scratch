import torch
from lib.layers import Module, Linear, RNN, LayerNorm
from lib.functions import init
from lib.functions.activations import softmax


class RNN_factory(Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, direction='forward', layer_norm=False, device='cpu'):
        assert direction in ('forward', 'backward', 'bidirectional'), "direction must be one of ('forward', 'backward', 'bidirectional')"
        self.hidden_size = hidden_size if direction != 'bidirectional' else hidden_size * 2
        self.n_layers = n_layers
        self.input_size = input_size
        self.direction = direction
        self.device = device

        self.rnn_layers = [RNN(input_size, hidden_size, backward=(direction == 'backward'), layer_norm=layer_norm, device=device) for _ in range(n_layers)]
        if direction == 'bidirectional':
            self.rnn_layers_reverse = [RNN(input_size, hidden_size, backward=True, layer_norm=layer_norm, device=device) for _ in range(n_layers)]

        # register the RNN layers (in the right order) as attributes to be detected by self.parameters()
        for i in range(n_layers):
            setattr(self, f'rnn_{i}', self.rnn_layers[i])
            if direction == 'bidirectional':
                setattr(self, f'rnn_{i}_rev', self.rnn_layers_reverse[i])

        self.out = Linear(self.hidden_size, output_size, device=device, weights_init=init.xavier_normal)

    def forward(self, x, h=None, logits=False):
        assert len(x.shape) == 2, 'x must be a 2D tensor (batch_size, time_steps)'
        if self.direction == 'bidirectional':
            h = h if h is not None else (None, None)
            assert len(h) == 2, 'For bi-directional RNN h must be a stack of two hidden states (one for each direction)'

        for i in range(self.n_layers):
            if self.direction != 'bidirectional':
                z, h = self.rnn_layers[i].forward(x, h)
            else:
                z_f, h_f = self.rnn_layers[i].forward(x, h[0])
                z_b, h_b = self.rnn_layers_reverse[i].forward(x, h[1])
                z = torch.concat((z_f, z_b), dim=-1)
                h = torch.stack((h_f, h_b))

        y = self.out.forward(z)
        if not logits:
            y = softmax(y)

        return y, h

    @torch.no_grad()
    def sample(self, n_samples=1, temperature=1., seed_seq=None):
        if seed_seq is None:
            x = torch.randint(0, self.input_size, size=(1,))
        else:
            x = torch.tensor(seed_seq, device=self.device)

        h = None
        seq = []
        for n in range(n_samples):
            z, h = self.forward(x.view(1, len(x)), h, logits=True)
            p = softmax(z[0][-1] / temperature)
            token = torch.multinomial(p, num_samples=1)  # sample single token
            seq.append(token.item())
            x = token

        return seq


class EchoStateNetwork(Module):

    def __init__(self, input_size, hidden_size, output_size, spectral_radius=2., sparsity=.90, device='cpu'):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

        self.rnn = RNN(input_size, hidden_size, device=device)
        self.out = Linear(hidden_size, output_size, device=device, weights_init=init.xavier_normal)

        # make all input-hidden and hidden-hidden parameters fixed, to be used as a reservoir
        for param in self.rnn.parameters(named=False):
            param.requires_grad = False

        # tune the spectral radius of the hidden-hidden weights
        Whh = self.rnn.cell.hidden.weight
        Whh *= torch.rand_like(Whh) > sparsity          # set 90% sparse connections:
        Whh /= torch.linalg.eigvals(Whh).abs().max()    # set the spectral radius of the hidden-hidden weights to 1
        Whh *= spectral_radius                          # scale up the spectral radius to 2, because the tanh saturation

    def forward(self, x, h=None, logits=False):
        assert len(x.shape) == 2, 'x must be a 2D tensor (batch_size, time_steps)'

        z, h = self.rnn.forward(x, h)
        y = self.out.forward(z)
        if not logits:
            y = softmax(y)

        return y, h

