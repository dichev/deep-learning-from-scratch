import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from collections import namedtuple

from models.recurrent_networks import Seq2Seq, RNN_factory
from lib.layers import Module, Linear, Embedding, RNN_cell, Conv2d, MaxPool2d, ReLU, Sequential, Dropout, Flatten, BatchNorm2d, AdditiveAttention, DotProductAttention
from lib.functions.activations import relu
from lib.functions import init
from preprocessing.transforms import batched_crop_on_different_positions
from utils import images as I


class RecurrentAttention(Module):
    """
    Paper: Recurrent Models of Visual Attention
    https://arxiv.org/pdf/1406.6247.pdf
    """

    def __init__(self, steps=6, focus_size=8, k_focus_patches=3, n_classes=10):
        self.focus_size = focus_size
        self.k = k_focus_patches
        self.steps = steps

        # Glimpse Network
        self.gnet_img = Linear(self.k * self.focus_size * self.focus_size, 128)
        self.gnet_loc = Linear(2, 128)
        self.gnet_combine = Linear(128*2, 256)

        # Core network
        self.rnn = RNN_cell(256, 512, use_relu=True)

        # Heads
        self.head_action = Linear(512, n_classes)
        self.head_loc = Linear(512, 2)

    def forward(self, x, loc=None):
        B, C, W, H = x.shape

        if loc is None:  # start in random location by default
            loc = torch.Tensor(B, 2).uniform_(-1, 1).to(x.device)

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



class SpatialTransformer(Module):
    """
    Paper: Spatial Transformer Networks
    https://arxiv.org/pdf/1506.02025.pdf
    """

    def __init__(self, transformation_mode='affine', localisation_net=None):
        self.mode = transformation_mode
        if self.mode == 'affine':       # [[a b t1], [c d t2]]
            n_params = 6
        elif self.mode == 'attention':  # [[s 0 t1], [0 s t2]]
            n_params = 3
        else:
            raise NotImplemented

        if localisation_net is None:
            self.localisation = Sequential(                                        # in:  1, 28, 28
                Conv2d(in_channels=1, out_channels=8, kernel_size=5),              # ->   8, 24, 24
                BatchNorm2d(8), ReLU(),
                MaxPool2d(kernel_size=2, stride=2),                                # ->   8, 12, 12
                Conv2d(in_channels=8, out_channels=16, kernel_size=5),             # ->  16,  8,  8
                BatchNorm2d(16), ReLU(),
                MaxPool2d(kernel_size=2, stride=2),                                # ->  16,  4,  4
                Flatten(),                                                         # -> 256
                Linear(input_size=16*4*4, output_size=64), ReLU(),                 # ->  64
                Linear(input_size=64, output_size=n_params)                        # -> n_params
            )
        else:
            self.localisation = localisation_net

        # Initialize as identity transformation
        with torch.no_grad():
            self.localisation[-1].weight.data.zero_()
            if self.mode == 'affine':
                self.localisation[-1].bias.data.copy_(torch.eye(2, 3, dtype=torch.float).flatten())
            elif self.mode == 'attention':
                self.localisation[-1].bias.data.copy_(torch.tensor([1, 0, 0]))

    def forward(self, U):
        B, C, H, W = U.shape

        # Localisation net - predict transformation over input image
        theta = self.localisation.forward(U)               # (B, n_params)
        transform = self.to_transformation_matrix(theta)   # (B, 2, 3) <- affine transformation matrix

        # Grid generator: transform the input image grid (with reverse mapping)
        grid_source = I.affine_grid(transform, U.size())

        # Sampler: sample input pixel values after the transformation with bilinear interpolation
        x_sampled = I.transform_image(U, grid_source, interpolation='bilinear')

        return x_sampled

    def to_transformation_matrix(self, theta):
        B = len(theta)
        if self.mode == 'affine':         # [[a b t1], [c d t2]]
            return theta.view(B, 2, 3)
        elif self.mode == 'attention':    # [[s 0 t1], [0 s t2]]
            s, t1, t2 = theta.unbind(-1)
            transform = torch.zeros(B, 2, 3, device=theta.device, dtype=torch.float)
            transform[:, 0, 0] = transform[:, 1, 1] = s         # dilation (scaling in same aspect ratio)
            transform[:, 0, 2], transform[:, 1, 2] = t1, t2     # translation
            return transform

    @torch.no_grad()
    def visualize(self, X, y, samples_per_digit=6, title=''):
        N = samples_per_digit
        X = torch.cat([X[y == i][:N] for i in range(10)], dim=0)  # filter and sort by digit
        X_transformed = self.forward(X)

        img_grid = make_grid(X.cpu(), padding=1, pad_value=.5, nrow=N).permute(1, 2, 0)
        img_grid_transformed = make_grid(X_transformed.cpu(), padding=1, pad_value=.5, nrow=N).permute(1, 2, 0)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_grid); ax1.axis(False); ax1.set_title('Dataset')
        ax2.imshow(img_grid_transformed); ax2.axis(False); ax2.set_title('Transformed')
        plt.suptitle(title); plt.tight_layout(); plt.show()


class SpatialTransformerNet(Module):

    def __init__(self, n_classes=10, transformation_mode='affine'):
        self.spatial_transform = SpatialTransformer(transformation_mode)

        self.body = Sequential(                                                # in:   1, 28, 28
            Conv2d(in_channels=1, out_channels=8, kernel_size=5),              # ->    8, 24, 24
            BatchNorm2d(8), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),                                # ->    8, 12, 12
            Conv2d(in_channels=8, out_channels=16, kernel_size=5),             # ->   16,  8,  8
            BatchNorm2d(16), ReLU(),
            MaxPool2d(kernel_size=2, stride=2),                                # ->   16,  4,  4
        )
        self.head = Sequential(
            Flatten(),                                                         # ->  256
            # Dropout(.5),                                                     # ->  64
            Linear(input_size=16*4*4, output_size=64), ReLU(),                 # ->  64
            Linear(input_size=64, output_size=n_classes)                       # ->  n_classes
        )

    def forward(self, x):
        x = self.spatial_transform.forward(x)
        x = self.body.forward(x)
        x = self.head.forward(x)
        return x


Context = namedtuple('Context', ['state', 'enc_outputs', 'attn_pad_mask'])


class AttentionEncoder(Module):
    """
    Paper: Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(self, vocab_size, embed_size, hidden_size, cell='gru', n_layers=4, layer_norm=False, padding_idx=0):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.rnn = RNN_factory(embed_size, hidden_size, cell, n_layers, direction='forward', layer_norm=layer_norm)
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

    def forward(self, x):
        batch_size, seq_len = x.shape
        pad_mask = (x != self.padding_idx)

        x = self.emb.forward(x)                     # (B, T) -> (B, T, embed_size)
        enc_out, enc_states = self.rnn.forward(x)   # (B, T, embed_size)) -> (B, T, hidden_size), [h, C]

        return enc_out, Context(enc_states, enc_out, pad_mask)



class AdditiveAttentionDecoder(Module):
    """
    Paper: Neural Machine Translation by Jointly Learning to Align and Translate
    https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size, attn_hidden_size, attn_dropout=0., cell='gru', n_layers=4, padding_idx=0):
        self.emb = Embedding(vocab_size, embed_size, padding_idx)
        self.attn_additive = AdditiveAttention(query_size=enc_hidden_size, key_size=enc_hidden_size, hidden_size=attn_hidden_size, dropout=attn_dropout)
        self.rnn = RNN_factory(embed_size + hidden_size, hidden_size, cell, n_layers, direction='forward', layer_norm=False)
        self.out = Linear(hidden_size, vocab_size, weights_init=init.xavier_normal_)
        self.hidden_size = hidden_size

    def forward(self, x, context: Context):
        B, T = x.shape
        x = self.emb.forward(x)                                       # (B, T) -> (B, T, emb_dec)

        output = []
        states, enc_outputs, attn_pad_mask = context
        for t in range(T):
            # Collect attention features
            h = states[0][-1]                                         # (B, emb_h)     <-  [h=(n_layers, B, emb_h), cell]  using only the hidden state from the !LAST! layer of the encoder
            query = h.unsqueeze(1)                                    # (B, 1, emb_h)  <-  that is a single query for each batch item
            attn_features, attn_weights = self.attn_additive.forward(query, key=enc_outputs, value=enc_outputs, attn_mask=attn_pad_mask.unsqueeze(1))

            # Decode from the combined input and attention features
            x_t = x[:, t:t+1]
            x_cat = torch.cat((attn_features, x_t), dim=-1)   # (B, 1, emb_dec + emb_val)
            out, states = self.rnn.forward(x_cat, states)             # (B, t, hidden_size), [h, C]
            output.append(out)

        output = torch.concat(output, dim=1)
        y = self.out.forward(output)

        return y, Context(states, context.enc_outputs, context.attn_pad_mask)
