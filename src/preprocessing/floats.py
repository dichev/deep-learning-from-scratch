
def normalizeMinMax(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min().item()
    if x_max is None:
        x_max = x.max().item()

    diff = (x_max - x_min) or 1. # avoid division by zero if all values are the same
    x = (x - x_min) / diff

    if ((x < 0.) | (x > 1.)).any():
        print(f'WARNING - The normalized data is using {x_min=}, {x_max=} and have values outside the [0, 1] range: x={x[(x < 0.) | (x > 1.)][:10]}.. They will be clipped!')
        x = x.clamp(x_min, x_max)

    return x


