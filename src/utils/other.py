def nested(t):
    for i in t:
        if isinstance(i, tuple):
            yield from nested(i)
        else:
            yield i

# tup = ((1, 2, 3), (4, 5, (6, 7)), 8, 9)
# for elem in nested(tup):
#     print(elem)


def conv2d_calc_out_size(X, kernel_size, stride=1, padding=0, dilation=1):
    N, C, W, H, = X.shape
    assert W == H, f'Expected square images as input, but got {W}x{H}'

    size = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    if size != int(size):
        print(f'Caution: The expected output size ({size:.1f}x{size:.1f}) is not an integer. Consider adjusting stride/padding/kernel to get an integer output size. Using rounded value: {int(size)}x{int(size)}')
    return int(size)

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
