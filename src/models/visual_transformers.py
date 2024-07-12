import torch
from lib.layers import Param, Module, Sequential, Linear, Embedding, LayerNorm, Dropout, ModuleList, PatchEmbedding, Conv2d, BatchNorm2d, ReLU
from models.transformer_networks import TransformerEncoderLayer


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
        B, C, W, H = X.shape
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

