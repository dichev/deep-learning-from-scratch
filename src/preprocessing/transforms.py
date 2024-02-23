import torch


def random_canvas_expand(img, width=60, height=60):
    C, H, W = img.shape
    assert H <= height and W <= width
    out = torch.zeros((C, height, width), device=img.device)
    y = torch.randint(height - H + 1, (1,))
    x = torch.randint(width - W + 1, (1,))
    out[:, y:y+H, x:x+W] = img
    return out


def batched_crop_on_different_positions(imgs, coords, size):
    """ Vectorized version of:
    B = len(imgs)
    crops = torch.zeros(B, size, size, device=imgs.device)
    for b in range(B):
        x, y = coords[b, 0], coords[b, 1]
        crops[b] = imgs[b, y:y + size, x:x + size]
    """
    B = len(imgs)

    offset = torch.arange(size, device=imgs.device)
    offset_x, offset_y = torch.meshgrid(offset, offset, indexing='xy')

    # Generate indices for each pixel of each crop
    x, y = coords[:, 0], coords[:, 1]
    x_indices = x.view(B, 1, 1) + offset_x
    y_indices = y.view(B, 1, 1) + offset_y
    batch_indices = torch.arange(B, device=imgs.device).view(B, 1, 1).expand(B, size, size)

    # Extract patches
    crops = imgs[batch_indices, y_indices, x_indices]

    return crops
