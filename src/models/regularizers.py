def L2_norm(parameters, lambd=1):
    norm = 0
    for name, param in parameters:
        norm += lambd * (param**2).sum()
    return norm

