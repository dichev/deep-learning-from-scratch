import torch
from tqdm import trange
from data.mnist import MNIST
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torchvision.transforms import v2 as transforms

from models.diffusion_models import DenoiseDiffusion
from models.residual_networks import UNet_DDPM
from lib.optimizers import AdamW
from lib.functions.losses import mse_loss
from utils.rng import seed_global
from preprocessing.integer import one_hot


# Hyperparams & settings
seed_global(2)
EPOCHS = 50
LEARN_RATE = 2e-4
BATCH_SIZE = 128
T = 1000                # diffusion steps
BETAS = (1e-4, 0.02)    # variance linear schedule
img_sizes = (3, 32, 32)
DEVICE = 'cuda'


# Data loaders
train_loader, val_loader, test_loader = MNIST(batch_size=BATCH_SIZE, train_val_split=(1., 0.), transforms = [
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),                     # normalize to [ 0, 1]
    transforms.Pad(2),                                                 # 28x28 -> 32x32 (for nicer up/down sampling by the U-net)
    # transforms.Normalize((0.5,), (0.5,)),                            # normalize to [-1, 1]  <- DISABLED for MNIST: speeds up training by ignoring the background pixels data (i.e. all 0 values)
    transforms.Lambda(lambda x: x * (torch.rand(3, 1, 1)*.6 +.4)),     # add colors to the digits (1, H, W) -> (3, H, W)
])
X_sample, y_sample = next(iter(train_loader))  # non-varying sample used for visualizations
assert X_sample.shape == (BATCH_SIZE, *img_sizes)


# Model
predictor = UNet_DDPM(img_sizes, context_features=10, max_timesteps=T + 1).to(DEVICE)
model = DenoiseDiffusion(img_sizes, T=T, noise_predictor=predictor, betas=BETAS).to(DEVICE)
optimizer = AdamW(predictor.parameters(), LEARN_RATE)


# Training
for epoch in range(EPOCHS):
    avg_loss = 0
    pbar = trange(len(train_loader), desc=f'Epoch {epoch+1}/{EPOCHS}')
    for x0, y in train_loader:
        x0 = x0.to(DEVICE)
        batch_size = x0.shape[0]  # might be less than BATCH_SIZE
        t = torch.randint(1, T+1, (batch_size, )).to(DEVICE)
        context = one_hot(y, num_classes=10).to(DEVICE)

        optimizer.zero_grad()
        x_t, noise = model.diffuse(x0, t)
        noise_pred = predictor(x_t, t, context)
        loss = mse_loss(noise_pred, noise)  # simplified (ignoring the schedule weighting) variational bound (ELBO)
        avg_loss = .9 * avg_loss + .1 * loss if avg_loss > 0 else loss
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_postfix_str(f'{loss=:.4f} {avg_loss=:.4f}')
    pbar.close()


    # Generate some samples
    n = 10
    C, H, W = img_sizes
    digits = torch.arange(n)
    print(f'Generating {n} images with context {digits}..')
    x_t, history = model.sample_denoise(n, context=one_hot(digits, num_classes=n), device=DEVICE)

    # Visualize the backward diffusion process
    steps = torch.arange(n+1) * 100  # on each 100 diffusion steps in [0, T]
    x_seq = history[steps]
    grid = make_grid(x_seq.transpose(0,1).reshape(n*len(steps), C, H, W), nrow=n+1, padding=2).permute(1, 2, 0).detach().cpu()
    plt.figure(figsize=(7, 6))
    plt.imshow(grid)
    plt.xticks(ticks=W//2 + steps / (steps[-1]/(grid.shape[0])), labels=['$t_{'+str(t)+'}$' for t in steps.tolist()])
    plt.yticks(ticks=H//2 + digits*(H+2) , labels=[f'ctx={d}' for d in digits.tolist()])
    plt.title(f'Epoch {epoch+1}/{EPOCHS} | loss={avg_loss.item():.4f}')
    plt.tight_layout()
    plt.show()
