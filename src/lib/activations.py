import numpy as np

def sign(x):
    return np.where(x >= 0, 1, -1)

def unit_step(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))