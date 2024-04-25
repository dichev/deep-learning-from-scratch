import pytest
import torch
from lib import optimizers

def train_loop(optim, param, prefix='', steps=10):
    for step in range(steps):
        optim.zero_grad()
        loss = (param ** 2).sum()
        loss.backward()
        optim.step()
        print(prefix + f"#{step} - Loss: {loss.item():>8.4f}", f"Params norm {param.norm():.4f}, {param.detach()}")
        yield param


@pytest.mark.parametrize('weight_decay',  [0, 0.8])
def test_SGD(weight_decay):
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.SGD([paramA], lr=.1, weight_decay=weight_decay)
    optimizerB = optimizers.SGD([('param1', paramB)], lr=.1, weight_decay=weight_decay)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)


@pytest.mark.parametrize('momentum',  [0, 0.2])
def test_SGD_momentum(momentum):
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.SGD([paramA], lr=.1, momentum=momentum)
    optimizerB = optimizers.SGD_Momentum([('param1', paramB)], lr=.1, momentum=momentum)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)


def test_AdaGrad():
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.Adagrad([paramA], lr=.1, eps=1e-8)
    optimizerB = optimizers.AdaGrad([('param1', paramB)], lr=.1, eps=1e-8)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)


@pytest.mark.parametrize('decay',  [0, 0.9])
def test_RMSProp(decay):
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.RMSprop([paramA], lr=0.01, alpha=decay, eps=1e-08)
    optimizerB = optimizers.RMSProp([('param1', paramB)], lr=0.01, decay=decay, eps=1e-08)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)



def test_AdaDelta():
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.Adagrad([paramA], lr=.1)
    optimizerB = optimizers.AdaGrad([('param1', paramB)], lr=.1)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)


@pytest.mark.parametrize('decay',  [0, 0.9])
def test_AdaDelta(decay):
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.Adadelta([paramA], lr=1, rho=decay, eps=1e-06)
    optimizerB = optimizers.AdaDelta([('param1', paramB)], lr=None, decay=decay, eps=1e-06)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)


@pytest.mark.parametrize('decay',  [0, 0.99])
@pytest.mark.parametrize('momentum',  [0, 0.9])
def test_Adam(decay, momentum):
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.Adam([paramA], lr=.1, eps=1e-8, betas=(momentum, decay))
    optimizerB = optimizers.Adam([('param1', paramB)], lr=.1, eps=1e-8, momentum=momentum, decay=decay)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)


@pytest.mark.parametrize('decay',  [0, 0.99])
@pytest.mark.parametrize('momentum',  [0, 0.9])
@pytest.mark.parametrize('weight_decay',  [0, 0.1])
def test_AdamW(decay, momentum, weight_decay):
    paramA = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)
    paramB = torch.tensor([5.0, 0.0001, -4.2, 0], requires_grad=True)

    optimizerA = torch.optim.AdamW([paramA], lr=.1, weight_decay=weight_decay, eps=1e-8, betas=(momentum, decay))
    optimizerB = optimizers.AdamW([('param1', paramB)], lr=.1, weight_decay=weight_decay, eps=1e-8, momentum=momentum, decay=decay)

    for A, B in zip(train_loop(optimizerA, paramA, '(expect) '), train_loop(optimizerB, paramB, '(actual) ')):
        assert torch.allclose(A, B)

