
def normalizeMinMax(x, dim=-1):
    x_min, x_max = x.min(dim, keepdim=True)[0],  x.max(dim, keepdim=True)[0]
    diff = x_max - x_min
    diff[diff == 0] = 1.0  # avoid division by zero when all the elements in the vector are equal (e.g. x=[2,2,2])
    return (x - x_min) / diff

