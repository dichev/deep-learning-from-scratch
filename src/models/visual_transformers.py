import torch
import einops as ein
from lib.layers import Param, Module, Sequential, Linear, Embedding, LayerNorm, Dropout, ModuleList, PatchEmbedding, Conv2d, BatchNorm2d, ReLU, RelativeWindowAttention
from models.transformer_networks import TransformerEncoderLayer
from utils.images import window_partition, window_reverse


class VisionTransformer(Module):  # ViT
    """
    Paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://arxiv.org/pdf/2010.11929
    """

    def __init__(self, n_classes, img_size=224, patch_size=16, in_channels=3, embed_size=768, hidden_size=4*768, n_layers=12, attn_heads=8, dropout=0.1):
        assert img_size % patch_size == 0, f'Image size({img_size}) must be divisible by patch size({patch_size})'
        self.n_patches = (img_size // patch_size)**2
        self.max_seq_len = self.n_patches + 1  # +1 for the cls token

        self.cls_emb = Param((1, 1, embed_size))  # as in BERT
        self.emb = PatchEmbedding(patch_size, embed_size, in_channels)  # aka patchify stem
        self.pos_emb = Embedding(self.max_seq_len, embed_size)
        self.dropout = Dropout(dropout)
        self.layers = ModuleList([
            TransformerEncoderLayer(embed_size, hidden_size, attn_heads, dropout, norm_first=True, gelu_activation=True) for _ in range(n_layers)  # note: in the paper there two dropouts (after each ff linear layer)
        ])
        self.final_norm = LayerNorm(embed_size)
        self.out = Linear(embed_size, n_classes)

        self.patch_size, self.img_size, self.in_channels = patch_size, img_size, in_channels
        self.embed_size, self.hidden_size, self.n_layers, self.attn_heads = embed_size, hidden_size, n_layers, attn_heads
        self.init_parameters()

    @torch.no_grad()
    def init_parameters(self):
        self.cls_emb.data.zero_()
        self.pos_emb.weight.data.normal_(std=.02)

    def forward(self, X, flash=False):
        B, C, H, W = X.shape
        T = self.n_patches
        positions = torch.arange(T+1, device=X.device)

        # Embeddings
        cls = self.cls_emb.expand(B, -1, -1).to(X.device)
        x = self.emb(X)                             # (B, T, emb)
        x = torch.cat((cls, x), dim=1)      # (B, T+1, emb)
        x = x + self.pos_emb(positions)             # (B, T+1, emb)  <- (B, T+1, emb) + (T+1, emb)
        x = self.dropout(x)

        # Transformers
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask=None, flash=flash)
        x = self.final_norm(x)                      # (B, T+1, emb)

        # Classify only by the class token:
        y = self.out(x[:, 0])                       # (B, n_classes)
        return y

    def get_last_attn_weights(self):
        attn_weights = torch.stack([layer.attn.get_last_attn_weights() for layer in self.layers], dim=1)
        return attn_weights  # (b, n_layers, n_heads, t, t)



class VisionTransformerConvStem(VisionTransformer):
    """
    Paper: Early Convolutions Help Transformers See Better
    https://arxiv.org/pdf/2106.14881
    """

    def __init__(self, n_classes, img_size=224, patch_size=16, in_channels=3, embed_size=768, hidden_size=4*768, n_layers=12-1, attn_heads=8, dropout=0.1):
        super().__init__(n_classes, img_size, patch_size, in_channels, embed_size, hidden_size, n_layers, attn_heads, dropout)
        assert img_size == 224 and patch_size == 16, 'Expected img_size=224 and patch_size=16'
        C, E = in_channels, embed_size

        # Replace the projected patches with a convolutional stem (note there  should be one less transformer layer to compensate for the computation cost):
        self.emb = Sequential(                                                                                 # in:   3, 224, 224
            Conv2d(in_channels=C,    out_channels=E//8, kernel_size=3, stride=2, padding=1, bias=False),       # ->   48, 112, 112
            BatchNorm2d(E//8), ReLU(),
            Conv2d(in_channels=E//8, out_channels=E//4, kernel_size=3, stride=2, padding=1, bias=False),       # ->   96,  56,  56
            BatchNorm2d(E//4), ReLU(),
            Conv2d(in_channels=E//4, out_channels=E//2, kernel_size=3, stride=2, padding=1, bias=False),       # ->  192,  28,  28
            BatchNorm2d(E//2), ReLU(),
            Conv2d(in_channels=E//2, out_channels=E,    kernel_size=3, stride=2, padding=1, bias=False),       # ->  384,  14,  14
            BatchNorm2d(E), ReLU(),
            Conv2d(in_channels=E,    out_channels=E,    kernel_size=1, stride=1, padding=0, bias=True),        # ->  384,  14,  14
            lambda x: x.flatten(start_dim=2).mT                                                                # ->  14*14, 384      # <- matching to ViT's patchify stem
        )



class SwinTransformerBlock(Module):
    def __init__(self, embed_size, hidden_size, attn_heads, img_size, window_size=7, dropout=0.):
        self.window_transformer = TransformerEncoderLayer(embed_size, hidden_size, attn_heads, dropout,
                                                          norm_first=True, gelu_activation=True,
                                                          attn_layer=RelativeWindowAttention(embed_size, attn_heads, window_size, dropout))
        self.shifted_window_transformer = TransformerEncoderLayer(embed_size, hidden_size, attn_heads, dropout,
                                                                  norm_first=True, gelu_activation=True,
                                                                  attn_layer=RelativeWindowAttention(embed_size, attn_heads, window_size, dropout))
        self.img_size = img_size
        self.window_size = window_size
        self.shift_size = self.window_size // 2
        self.cached_attn_mask, _ = self.generate_shifted_attn_masks()
        # todo relative positions

    def generate_shifted_attn_masks(self):
        H = W = self.img_size
        M = self.window_size
        N = H // M * W // M

        # Generate tiles for each window, with specific pattern for the shifted edges (i.e. A, B, C regions from the paper)
        img_zones = torch.arange(N).view(H // M, W // M).repeat_interleave(M, dim=0).repeat_interleave(M, dim=1)
        img_zones[:, -self.shift_size:] += N
        img_zones[-self.shift_size:] += N*2

        # Generate attention mask for each window respecting the shifted patterns
        patterns = window_partition(img_zones.view(1, 1, H, W), window_size=M).view(N, M * M)
        attn_mask = patterns.unsqueeze(1) != patterns.unsqueeze(2)  # compare values along repeated rows and cols

        return attn_mask, img_zones  # N, M*M, M*M

    def forward(self, x, flash=False):
        B, C, H, W = x.shape
        M = self.window_size
        N = H // M * W // M
        shift_size = self.shift_size

        # Self-attention using regular non-overlapped windows
        x = window_partition(x, window_size=M)                                           # B, N, C, M, M
        x = ein.rearrange(x, 'b n c m1 m2 -> (b n) (m1 m2) c')                   # B, T, C       (as tokens)
        x = self.window_transformer(x, attn_mask=None, flash=flash)
        x = ein.rearrange(x, '(b n) (m1 m2) c -> b n c m1 m2', b=B, m1=M, m2=M)  # B, C, H, W
        x = window_reverse(x, window_size=M, height=H, width=W)                          # B, C, H, W

        # Self-attention using shifted window partitioning
        x = x.roll((-shift_size, -shift_size), dims=(-2, -1))                            # B, C, H, W    (cyclic shift on H, W for efficient batching)+
        x = window_partition(x, window_size=M)                                           # B, N, C, M, M
        x = ein.rearrange(x, 'b n c m1 m2 -> (b n) (m1 m2) c')                   # B, T, C       (as tokens)
        attn_mask = self.cached_attn_mask.unsqueeze(0).repeat_interleave(B, dim=0).view(B*N, 1, M*M, M*M)  # 1 is for n_attn_heads
        x = self.shifted_window_transformer(x, attn_mask=attn_mask.to(x.device), flash=flash)
        x = ein.rearrange(x, '(b n) (m1 m2) c -> b n c m1 m2', b=B, m1=M, m2=M)  # B, C, H, W
        x = window_reverse(x, window_size=M, height=H, width=W)                          # B, C, H, W
        x = x.roll((shift_size, shift_size), dims=(-2, -1))

        return x


class SwinTransformer(Module):  # Shifted windows transformer (Swin-T version)
    """
    Paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    https://arxiv.org/pdf/2103.14030
    """

    def __init__(self, n_classes, img_size=224, patch_size=4, window_size=7, in_channels=3, embed_size=96, hidden_size=4*768, n_layers=12, attn_heads=8, dropout=0.1):
        assert img_size % patch_size == 0, f'Image size({img_size}) must be divisible by patch size({patch_size})'
        self.n_patches = (img_size // patch_size)**2
        C = embed_size

        # in contrast to the paper implementation, here the patches are processed on their img dim (B, C, H, W) through the layers
        self.layers = ModuleList([

            # stage1
            PatchEmbedding(patch_size=4, in_channels=3, embed_size=C, keep_img_dim=True),      # 3, 224, 224  -> C, 56, 56    # todo they apply norm to patch emb / patch merge
            [SwinTransformerBlock(embed_size=C, hidden_size=4*C, attn_heads=C//32, img_size=56, window_size=7) for _ in range(6)],  # todo: what are the params? dropout?

            # stage2
            PatchEmbedding(patch_size=2, in_channels=C, embed_size=2*C, keep_img_dim=True),    # C, 56, 56   ->  2C, 28, 28
            [SwinTransformerBlock(embed_size=2*C, hidden_size=4*2*C, attn_heads=2*C//32, img_size=28, window_size=7) for _ in range(2)],

            # stage3
            PatchEmbedding(patch_size=2, in_channels=2*C, embed_size=4*C, keep_img_dim=True),  # 2C, 28, 28  ->  4C, 14, 14
            [SwinTransformerBlock(embed_size=4*C, hidden_size=4*4*C, attn_heads=4*C//32, img_size=14, window_size=7) for _ in range(6)],

            # stage4
            PatchEmbedding(patch_size=2, in_channels=4*C, embed_size=8*C, keep_img_dim=True),  # 4C, 14, 14  -> 8C, 7, 7
            [SwinTransformerBlock(embed_size=8*C, hidden_size=4*8*C, attn_heads=8*C//32, img_size=7, window_size=7) for _ in range(2)],
        ])
        self.final_norm = LayerNorm(8*C)
        self.out = Linear(8*C, n_classes)

        self.patch_size, self.window_size, self.img_size, self.in_channels = patch_size, window_size, img_size, in_channels
        self.embed_size, self.hidden_size, self.n_layers, self.attn_heads = embed_size, hidden_size, n_layers, attn_heads


    def forward(self, X, flash=False):
        B, C, H, W = X.shape
        T = self.n_patches

        # Transformers
        for i, layer in enumerate(self.layers):
            X = layer(X)  # todo: , flash=flash

        x = X.flatten(start_dim=2).mT   # B, C, H, W   ->  B, C, HW  ->  B, T, C
        x = self.final_norm(x)                      # (B, T, emb)

        # Classify only by the class token:
        x = x.mean(dim=1) # todo use avg pool
        y = self.out(x)                       # (B, n_classes)
        return y
