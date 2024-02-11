import torch
import warnings


def identity(n, sparse=False, device=None):
    if sparse:
        indices = torch.arange(n).unsqueeze(0).repeat(2, 1)
        values = torch.ones(n)
        return torch.sparse_coo_tensor(indices, values, (n, n), device=device)
    else:
        return torch.eye(n, device=device)


def conv2d_calc_out_size(X, kernel_size, stride=1, padding=0, dilation=1):
    N, C, W, H, = X.shape

    if isinstance(padding, tuple):
        pad_left, pad_right, pad_top, pad_bottom = padding
    else:
        pad_left = pad_right = pad_top = pad_bottom = padding

    width  = (W + (pad_left + pad_right) - dilation * (kernel_size - 1) - 1) / stride + 1
    height = (H + (pad_top + pad_bottom) - dilation * (kernel_size - 1) - 1) / stride + 1
    if width != int(width) or height != int(height):
        warnings.warn(f'Caution: Input{list(X.shape)} - The expected output size after convolution ({width:.1f}x{height:.1f}) is not an integer. Consider adjusting stride/padding/kernel to get an integer output size. Using rounded value: {int(width)}x{int(height)}')
    return int(width), int(height)

def conv2d_pad_string_to_int(padding, kernel_size):
    if padding == 'valid':
        return 0
    elif padding == 'same':
        assert kernel_size % 2 == 1, f'For "same" padding the kernel_size is expected to be odd number, but got: {kernel_size}'
        return (kernel_size - 1) // 2
    elif padding == 'full':
        return kernel_size - 1
    else:
        return padding
