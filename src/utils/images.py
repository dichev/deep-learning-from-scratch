import torch
import torch.nn.functional as F


def affine_grid(transform_matrix, out_size):
    B, _, H, W = out_size
    device = transform_matrix.device

    xx, yy = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='xy')
    grid = torch.stack((xx, yy, torch.ones(H, W)), dim=-1)
    grid = grid.view(1, H, W, 3).to(device)

    grid_source = torch.einsum('bij,bhwj->bhwi', transform_matrix, grid)  # (grid.view(1, H*W, 3) @ theta.permute(0, 2, 1)).view(B, H, W, 2)

    return grid_source  # note the grid coordinates are normalized to [-1, 1]


def denormalize_grid(grid):  # grid specifies the sampling pixel locations normalized by the input spatial dimensions  in the range of [-1, 1].
    B, H, W, _ = grid.shape
    return 0.5 * (grid + 1.0) * torch.tensor([W - 1, H - 1], device=grid.device)


def transform_image(img, grid, interpolation='bilinear'):  # that's a differentiable interpolation
    assert interpolation == 'bilinear', f'Only bilinear interpolation is supported, but got {interpolation}'
    B, C, H, W = img.shape

    # Get denormalized grid coordinates
    x, y = denormalize_grid(grid).unbind(-1)

    # Zero pad (i.e. border) the image and shift respectively the source grid, to handle interpolation on the edges
    pad = 1
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='constant', value=0)
    x = x + pad
    y = y + pad

    # Compute the 4 surrounding image pixels and clamp the coordinates outside the padded image boundaries
    x = x.clamp(0, W - 1 + 2 * pad)  # Because the padding all coordinates outside the output size will be corresponding to zero pixels
    y = y.clamp(0, H - 1 + 2 * pad)
    x0, y0 = x.floor(), y.floor()
    x1, y1 = x0 + 1, y0 + 1
    x1 = x1.clamp(0, W - 1 + 2 * pad)
    y1 = y1.clamp(0, H - 1 + 2 * pad)

    # Bilinear interpolation:
    # 1. compute the weights
    dx = (x - x0)  # / (x1 - x0)
    dy = (y - y0)  # / (y1 - y0)
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy

    # 2. expand the channel dimension to perform the same transformation on each channel
    w00, w10, w01, w11 = [w.view(B, 1, H, W).expand(B, C, H, W) for w in (w00, w10, w01, w11)]
    x0, x1, y0, y1 = [x.view(B, 1, H, W).expand(B, C, H, W).long() for x in (x0, x1, y0, y1)]

    # 3. sample the get pixel values with indices
    batch_indices = torch.arange(B).view(B, 1, 1, 1)
    channel_indices = torch.arange(C).view(1, C, 1, 1)
    v00 = img_pad[batch_indices, channel_indices, y0, x0]
    v10 = img_pad[batch_indices, channel_indices, y0, x1]
    v01 = img_pad[batch_indices, channel_indices, y1, x0]
    v11 = img_pad[batch_indices, channel_indices, y1, x1]

    # 4. finally interpolate the pixels
    weights = torch.stack((w00, w10, w01, w11))
    values = torch.stack((v00, v10, v01, v11))
    x_sampled = (weights * values).sum(dim=0)  # == w00*v00 + w10*v10 + w01*v01 + w11*v11

    return x_sampled


#
# import torch.nn.functional as F
# grid_check = F.affine_grid(theta, U.size(), align_corners=True)
# x_sampled_check = F.grid_sample(U, grid_check, align_corners=True, mode='bilinear', padding_mode='zeros')
# assert grid_source.shape == grid_check.shape and torch.allclose(grid_source, grid_check, rtol=1e-4, atol=1e-4)
# assert x_sampled_check.shape == x_sampled.shape and torch.allclose(x_sampled_check, x_sampled, rtol=1e-5, atol=1e-5)
#
#
# return x_sampled_check

# x = torch.ones(1, 1, 28, 30).float()
# y = SpatialTransformer(in_channels=x.shape[1], test=True).forward(x)
# x = torch.ones(2, 1, 28, 30).float()
# y = SpatialTransformer(in_channels=x.shape[1], test=True).forward(x)
# x = torch.ones(1, 2, 28, 30).float()
# y = SpatialTransformer(in_channels=x.shape[1], test=True).forward(x)
# x = torch.ones(3, 2, 28, 30).float()
# y = SpatialTransformer(in_channels=x.shape[1], test=True).forward(x)
#
# x = torch.arange(28*28).view(1,1,28,28).float()
# y = SpatialTransformer(in_channels=x.shape[1]).forward(x)
# # todo: test with non-identy theta
# theta = torch.tensor([[
#     [1.0, 0.0, 0.0000],
#     [0.0, 1.0, 0.0000],
# ]]) * 1.1
