import torch
from models.layers import Module, Linear, Embedding
from functions import init
from functions.activations import tanh, softmax


class RNN(Module): # layer

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


class UniRNN(Module):

    def __init__(self, input_size, hidden_size, output_size, device='cpu'):
        self.rnn = RNN(input_size, hidden_size, device=device)
        self.out = Linear(hidden_size, output_size, device=device, weights_init=init.xavier_normal)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, h0=None, logits=False):
        z, h = self.rnn.forward(x, h0)
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
        latex = self.rnn.expression() + '\n'
        latex += r'$y_t = softmax(W_{hy} h_t + b_y$)'
        return latex

    def __repr__(self):
        return f'UniRNN(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.out.output_size}): {self.n_params} params'


