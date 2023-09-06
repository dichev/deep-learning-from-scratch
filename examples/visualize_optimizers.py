import torch
import numpy as np
import matplotlib.pyplot as plt
from models import optimizers


# settings
lr = 0.8
max_iterations = 100


def f(x, y):
    a = 1/25
    b = 1
    return a*(x**2) + b*(y**2)

class Model:
    def __init__(self, x_start=10., y_start=10.):
        self.x = torch.tensor([x_start], requires_grad=True)
        self.y = torch.tensor([y_start], requires_grad=True)

    def forward(self):
        return f(self.x, self.y)


paths = []
for optimizer in [optimizers.SGD, optimizers.SGD_Momentum, optimizers.AdaGrad, optimizers.RMSProp, optimizers.AdaDelta, optimizers.Adam]:
    name = optimizer.__name__
    print('Optimizing with', name)
    model = Model()
    optimizer = optimizer([('x', model.x), ('y', model.y)], lr)

    path = [(model.x.item(), model.y.item())]
    for _ in range(max_iterations):
        y_hat = model.forward()
        y_hat.backward()
        optimizer.step().zero_grad()
        path.append((model.x.item(), model.y.item()))
        if y_hat < 0.01:
            break

    path = np.array(path)
    paths.append((name, path))


# Visualize the path of the optimizers:
x_grid = torch.linspace(-12, 12, 400)
y_grid = torch.linspace(-12, 12, 400)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
Z = f(X, Y)

plt.figure(figsize=(10, 10))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.title(f'Optimization path: lr={lr}')
for name, path in paths:
    pos = plt.scatter(*path[0], label=f'{name} - {len(path)} iterations')
    for i in range(1, len(path)):
        plt.arrow(path[i-1, 0], path[i-1, 1], path[i, 0]-path[i-1, 0], path[i, 1]-path[i-1, 1], color=pos.get_facecolor()[0], head_width=0.2, head_length=0.2, length_includes_head=True)
plt.legend()
plt.show()

