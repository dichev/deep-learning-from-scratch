import torch
from tqdm import trange
from data.mnist import MNIST
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from torchvision.transforms import v2 as transforms

from models.diffusion_models import DenoiseDiffusion
from models.residual_networks import UNet_simple
from lib.optimizers import AdamW
from lib.functions.losses import mse_loss
from utils.rng import seed_global


# Hyperparams & settings
seed_global(2)
EPOCHS = 50
LEARN_RATE = 2e-4
BATCH_SIZE = 128
T = 1000            # diffusion steps
BETAS = (1e-4, 0.02)  # variance linear schedule
DEVICE = 'cuda'


# Data loaders
train_loader, val_loader, test_loader = MNIST(batch_size=BATCH_SIZE, train_val_split=(1., 0.), transforms = [
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),    # normalize to [ 0, 1]
    transforms.Pad(2),                                # 28x28 -> 32x32 (for nicer up/down sampling by the U-net)
    transforms.Normalize((0.5,), (0.5,)),  # normalize to [-1, 1]
])
X_sample, y_sample = next(iter(test_loader))  # non-varying sample used for visualizations
T_sample = torch.randint(T, (BATCH_SIZE, ))


# Model
predictor = UNet_simple(img_sizes=(1, 32, 32), max_timesteps=T+1).to(DEVICE)
model = DenoiseDiffusion(img_size=32, T=T, noise_predictor=predictor, betas=BETAS).to(DEVICE)
optimizer = AdamW(predictor.parameters(), LEARN_RATE)


# Training
for epoch in range(EPOCHS):
    avg_loss =  0
    n = len(train_loader)
    pbar = trange(n, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for x0, _ in train_loader:  # todo: condition on the label
        x0 = x0.to(DEVICE)
        batch_size = x0.shape[0]  # might be less than BATCH_SIZE
        t = torch.randint(1, T+1, (batch_size, )).to(DEVICE)

        optimizer.zero_grad()
        x_t, noise = model.diffuse(x0, t)
        noise_pred = predictor(x_t, t)
        loss = mse_loss(noise_pred, noise)  # simplified (ignoring the schedule weighting) variational bound (ELBO)
        avg_loss = .9 * avg_loss + .1 * loss if avg_loss > 0 else loss
        loss.backward()
        optimizer.step()

        pbar.update()
        pbar.set_postfix_str(f'{loss=:.4f} {avg_loss=:.4f}')
    pbar.close()


    # Generate some samples
    print('Generating images..')
    x_t = model.sample_denoise(n=9, device=DEVICE)
    grid = make_grid(x_t, nrow=3).detach().cpu().permute(1, 2, 0)
    plt.imshow(grid, cmap='gray'); plt.title(f'Epoch {epoch+1}/{EPOCHS} | loss={avg_loss.item():.4f}'); plt.axis(False); plt.show()




