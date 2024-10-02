import torch
from math import sqrt


def sign(x):
    return torch.where(x >= 0, 1, -1)


def unit_step(x):
    return torch.where(x >= 0, 1, 0)


@torch.jit.script
def sigmoid(x): # numerically stable
    e = torch.exp(-x.abs())
    return torch.where(x >= 0, 1 / (1 + e), e / (1 + e))


@torch.jit.script
def tanh(x): # numerically stable (sign is faster than masking)
    e = torch.exp(-2 * x.abs())
    return sign(x) * (1 - e) / (1 + e)


def relu(x):
    return torch.clip(x, 0)


@torch.jit.script
def gelu(x):  # x * F(x), where F is the cumulative normal distribution
    return x * 0.5 * (1 + torch.erf(x / sqrt(2)))  # ≈ sigmoid(1.702 * x)


@torch.jit.script
def gelu_tanh_approx(x):
    return x * 0.5 * (1 + torch.tanh(sqrt(2/torch.pi) * (x + 0.044715 * x**3)))


def silu(x):
    return x * sigmoid(x)


def swish(x, beta=1.):
    """
    swish(x, beta=1)     = silu(x)
    swish(x, beta=1.702) ≈ gelu(x)
    swish(x, beta=inf)   = relu(x)
    """
    return x * sigmoid(beta * x)


def glu(x, gate=sigmoid):  # Gated Linear Unit (implemented as non-parameterized function, to allow the linear projections to be defined as separate modules in the model)
    assert x.shape[-1] % 2 == 0, f'Expected even dimension but got {x.shape}. GLU expect the input to be two concatenated linear projections a and b (with equal size).'
    a, b = x.chunk(2, dim=-1)
    return a * gate(b)

def swiglu(x):
    return glu(x, gate=swish)


def softmax(z, dim=-1, ignore_mask=None):
    if ignore_mask is not None:
        assert z.ndim == ignore_mask.ndim, f'Expecting mask with the same dimension as {z.shape} but got {ignore_mask.shape}'
        z = z.masked_fill(ignore_mask, -torch.inf)  # exclude the masked scores (i.e. right paddings) from the gradients, by making their softmax probability essentially zero (e^-inf -> 0)

    """ Shift with the max value to avoid numerical overflows:
    ->  softmax(z) =  e^{z_i} / sum e^z  * e^{-c}/e^{-c} = e^{z_i-c} / sum e^{z-c} = softmax(z-c)
    """
    e = torch.exp(z - z.max(dim, keepdim=True)[0])
    return e / e.sum(dim, keepdim=True)


def log_softmax(z, dim=-1, ignore_mask=None):
    if ignore_mask is not None:
        assert z.ndim == ignore_mask.ndim, f'Expecting mask with the same dimension as {z.shape} but got {ignore_mask.shape}'
        z = z.masked_fill(ignore_mask, -torch.inf)  # exclude the masked scores (i.e. right paddings) from the gradients, by making their softmax probability essentially zero (e^-inf -> 0)

    """ Division to subtraction & LogSumExp trick:
    ->  ln(softmax(z)) = ln( e^{z_i} / sum e^z) = z_i - ln(sum e^z)
    """
    return z - log_sum_exp(z, dim)


def log_sum_exp(z, dim=-1):
    """ LogSumExp trick for numerical stability:
    ->  ln(sum e^z) = ln(sum e^z e^c e^{-c}) = c - ln(sum e^{z-c})
    The best choice for c is the max value of z, because then the largest exponent will be 1:
    ->  max(e^{x-max(x)}) = e^0 = 1
    """
    c = z.max(dim, keepdim=True)[0]
    return c + torch.log(torch.exp(z-c).sum(dim, keepdim=True))



