import torch
from models.layers import Module, Linear, Embedding
from functions import init
from functions.activations import tanh, softmax


class RNN_layer(Module):  # layer

    def __init__(self, input_size, hidden_size, device='cpu'):
        self.embed = Embedding(input_size, hidden_size, device=device)  # no bias
        self.hidden = Linear(hidden_size, hidden_size, device=device, weights_init=init.xavier_normal)
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, h=None):  # todo: support one-hot/dense input
        N, T = x.shape

        if h is None:
            h = torch.zeros(self.hidden_size, device=self.device)

        z = torch.zeros(N, T, self.hidden_size, device=self.device)
        for t in range(T):
            xh = self.embed.forward(x[:, t])  # directly select the column embedding
            hh = self.hidden.forward(h)
            h = tanh(xh + hh)
            z[:, t] = h

        return z, h  # h == z[:, -1]  (i.e. the final hidden state for each batch element)

    def expression(self):
        latex = r'$h_t = \tanh(W_{xh} x + W_{hh} h_{t-1} + b_h)$' + '\n'
        return latex

    def __repr__(self):
        return f'RNN(input_size={self.embed.input_size}, hidden_size={self.hidden.output_size}): {self.n_params} params'


class RNN(Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional=False, device='cpu'):
        self.rnn_f = RNN_layer(input_size, hidden_size, device=device)
        if bidirectional:
            self.rnn_b = RNN_layer(input_size, hidden_size, device=device)
        self.out = Linear(hidden_size, output_size, device=device, weights_init=init.xavier_normal)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.bidirectional = bidirectional

    def forward(self, x, h=None, logits=False):
        assert len(x.shape) == 2, 'x must be a 2D tensor (batch_size, time_steps)'

        if not self.bidirectional:
            z, h = self.rnn_f.forward(x, h)
        else:
            if h is None:
                h = (None, None)
            else:
                assert len(h) == 2, 'For bi-directional RNN h must be a stack of two hidden states (one for each direction)'

            z_f, h_f = self.rnn_f.forward(x, h[0])
            z_b, h_b = self.rnn_b.forward(x.flip(dims=[1]), h[1])
            h = torch.stack((h_f, h_b))
            z = z_f + z_b

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

    def expression(self):
        if self.bidirectional:
            latex  = r'$h_t^{(f)} = \tanh(W_{xh}^{(f)} x + W_{hh}^{(f)} h_{t-1}^{(f)} + b_h^{(f)})$' + '\n'
            latex += r'$h_t^{(b)} = \tanh(W_{xh}^{(b)} x + W_{hh}^{(b)} h_{t+1}^{(b)} + b_h^{(b)})$' + '\n'
            latex += r'$y_t = softmax(W_{hy}^{(f)} h_t^{(f)} + W_{hy}^{(b)} h_t^{(b)} + b_y$)'
        else:
            latex  = r'$h_t = \tanh(W_{xh} x + W_{hh} h_{t-1} + b_h)$' + '\n'
            latex += r'$y_t = softmax(W_{hy} h_t + b_y$)'
        return latex

    def __repr__(self):
        return f'RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.out.output_size}, bidirectional={self.bidirectional}): {self.n_params} params'