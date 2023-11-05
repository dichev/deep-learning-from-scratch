import torch

def sign(x):
    return torch.where(x >= 0, 1, -1)

def unit_step(x):
    return torch.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    e = torch.exp(x)
    return (e - 1/e) / (e + 1/e)

def relu(x):
    return torch.clip(x, 0)

def softmax(z, dim=-1):
    """ Shift with the max value to avoid numerical overflows:
    ->  softmax(z) =  e^{z_i} / sum e^z  * e^{-c}/e^{-c} = e^{z_i-c} / sum e^{z-c} = softmax(z-c)
    """
    e = torch.exp(z - z.max(dim, keepdim=True)[0])
    return e / e.sum(dim, keepdim=True)

def log_softmax(z, dim=-1):
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


