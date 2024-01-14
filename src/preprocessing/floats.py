
def normalizeMinMax(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min().item()
    if x_max is None:
        x_max = x.max().item()

    diff = (x_max - x_min) or 1.  # avoid division by zero if all values are the same
    x = (x - x_min) / diff

    if ((x < 0.) | (x > 1.)).any():
        print(f'WARNING - The normalized data is using {x_min=}, {x_max=} and have values outside the [0, 1] range: x={x[(x < 0.) | (x > 1.)][:10]}.. They will be clipped!')
        x = x.clamp(x_min, x_max)

    return x


def img_normalize(x, permute_channel=True):
    if permute_channel:  # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
    x = normalizeMinMax(x.float(), 0, 255) # range [0,1]
    x = (x - .5) / .5  # shift and scale in the typical range (-1, 1)
    return x

def img_unnormalize(x, permute_channel=True):
    if permute_channel:  # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
    x = (x * .5) + .5              # range [0,1]
    x = (x * 255).round().to(int)  # range [0,255]
    return x

