import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF

from lib.layers import Module, Linear, RNN_cell
from lib.functions.activations import relu
from preprocessing.transforms import batched_crop_on_different_positions


class RecurrentAttention(Module):
    """
    Paper: Recurrent Models of Visual Attention
    https://arxiv.org/pdf/1406.6247.pdf
    """

    def __init__(self, steps=6, focus_size=8, k_focus_patches=3, n_classes=10, device='cpu'):
        self.focus_size = focus_size
        self.k = k_focus_patches
        self.steps = steps
        self.device = device

        # Glimpse Network
        self.gnet_img = Linear(self.k * self.focus_size * self.focus_size, 128, device=device)
        self.gnet_loc = Linear(2, 128, device=device)
        self.gnet_combine = Linear(128*2, 256, device=device)

        # Core network
        self.rnn = RNN_cell(256, 512, use_relu=True, device=device)

        # Heads
        self.head_action = Linear(512, n_classes, device=device)
        self.head_loc = Linear(512, 2, device=device)

    def forward(self, x, loc=None):
        B, C, W, H = x.shape

        if loc is None:  # start in random location by default
            loc = torch.Tensor(B, 2).uniform_(-1, 1).to(self.device)

        locs = []
        a, state = None, (None, None)
        for i in range(self.steps):
            # Glimpse Sensor
            patches = self.glimpse_sensor(x, loc, self.k)  # (B, patches, focus_size, focus_size)
            patches = patches.view(B, -1)                  # (B, features)

            # Glimpse network
            g = torch.concat((
                relu(self.gnet_img.forward(patches)),
                relu(self.gnet_loc.forward(loc)),
            ), dim=-1)
            g = relu(self.gnet_combine.forward(g))

            # Core network
            state = self.rnn.forward(g, state)
            h, C = state

            # Heads
            loc = self.head_loc.forward(h)
            # loc = torch.rand_like(loc)*2-1  # for comparison
            if i + 1 == self.steps:  # paper: for classification experiments made a classification decision only at the last timestep
                a = self.head_action.forward(h)

            locs.append(loc)

        return a, locs

    def glimpse_sensor(self, x, loc, k=3):
        B, C, W, H = x.shape
        assert W == H and C == 1, f'Supports only square images with single channel, but got: {x.shape}'
        assert loc.shape == (B, 2), f'Expected batched loc of shape (B, 2), but got: {loc.shape}'

        # paper: glimpse locations were encoded as real-valued (x, y) in the range [-1, 1]
        coords = self.denormalize_loc(loc, W)

        # Extract the largest patch on specified coords
        imgs = x.view(B, W, H)
        size = self.focus_size * 2 ** (k-1)
        imgs_padded = TF.pad(imgs, size // 2)
        patches_K = batched_crop_on_different_positions(imgs_padded, coords, size)

        # Collect all the k patches, and resize them on the same size (to simulate  different resolution)
        patches = torch.zeros(B, k, self.focus_size, self.focus_size, device=imgs.device)
        for i in range(k):
            size = self.focus_size * 2 ** i
            patch = TF.center_crop(patches_K, size)
            patch = TF.resize(patch, size=[self.focus_size, self.focus_size], antialias=True)
            patches[:, i] = patch

        return patches

    @staticmethod
    def denormalize_loc(loc, img_size):
        coords = 0.5 * ((loc + 1.0) * img_size)
        coords = coords.round().long().clamp(min=0, max=img_size)
        return coords

    def visualize(self, x, loc):
        B, C, H, W = x.shape
        patches = self.glimpse_sensor(x, loc)
        pos = self.denormalize_loc(loc, W)

        fig, ax = plt.subplots(B, 2, figsize=(2 * 2, B * 2))
        for i in range(B):
            grid = make_grid(patches[i].unsqueeze(1).cpu(), normalize=True, pad_value=1, padding=1).permute(1, 2, 0)
            ax[i, 0].imshow(x[i].view(H, W).cpu(), cmap='gray')
            ax[i, 0].axis(False)
            for k in range(1, self.k + 1):
                size = self.focus_size * 2 ** (k - 1)
                ax[i, 0].add_patch(Rectangle(pos[i].cpu() - size // 2, size, size, linewidth=1, edgecolor='r', facecolor='none'))
            ax[i, 1].imshow(grid)
            ax[i, 1].axis(False)
        plt.tight_layout()
        plt.show()
