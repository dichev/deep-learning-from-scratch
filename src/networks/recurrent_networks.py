import torch
from models.layers import Module, Linear, RNN
from functions import init
from functions.activations import softmax


class UniRNN(Module):
    def __init__(self, input_size, hidden_size, output_size, backward=False, device='cpu'):
        self.rnn = RNN(input_size, hidden_size, backward, device=device)
        self.out = Linear(hidden_size, output_size, device=device, weights_init=init.xavier_normal)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.backward = backward
        self.device = device

    def forward(self, x, h=None, logits=False):
        assert len(x.shape) == 2, 'x must be a 2D tensor (batch_size, time_steps)'

        z, h = self.rnn.forward(x, h)
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
        latex  = self.rnn.expression()
        latex += r'$y_t = softmax(W_{hy} h_t + b_y$)'
        return latex

    def __repr__(self):
        return f'{self.__class__.__name__}(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.out.output_size}, backward={self.backward}): {self.n_params} params'


class BiRNN(UniRNN):

    def __init__(self, input_size, hidden_size, output_size, device='cpu'):
        self.rnn_f = RNN(input_size, hidden_size, backward=False, device=device)
        self.rnn_b = RNN(input_size, hidden_size, backward=True, device=device)
        self.out = Linear(hidden_size*2, output_size, device=device, weights_init=init.xavier_normal)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device

    def forward(self, x, h=None, logits=False):
        assert len(x.shape) == 2, 'x must be a 2D tensor (batch_size, time_steps)'
        if h is None:
            h = (None, None)
        assert len(h) == 2, 'For bi-directional RNN h must be a stack of two hidden states (one for each direction)'

        z_f, h_f = self.rnn_f.forward(x, h[0])
        z_b, h_b = self.rnn_b.forward(x, h[1])
        z = torch.concat((z_f, z_b), dim=-1)
        h = torch.stack((h_f, h_b))

        y = self.out.forward(z)
        if not logits:
            y = softmax(y)

        return y, h

    def expression(self):
        latex  = r'$h_t^{(f)} = \tanh(W_{xh}^{(f)} x + W_{hh}^{(f)} h_{t-1}^{(f)} + b_h^{(f)})$' + '\n'
        latex += r'$h_t^{(b)} = \tanh(W_{xh}^{(b)} x + W_{hh}^{(b)} h_{t+1}^{(b)} + b_h^{(b)})$' + '\n'
        latex += r'$y_t = softmax(W_{hy}^{(f)} h_t^{(f)} + W_{hy}^{(b)} h_t^{(b)} + b_y$)'
        return latex
