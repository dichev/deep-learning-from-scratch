import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import os
import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from networks.energy_based_models import RestrictedBoltzmannMachine
from models.training import batches


DEVICE = 'cuda'
DATA_DIR = './data/BW_images/'
EPOCHS = 200
BATCH_SIZE = 6
LEARN_RATE = 0.1
K_EPOCHS = 50  # increase contrastive divergence steps by 1 after this many epochs


# Prepare images (data)
files = [DATA_DIR+f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
images = np.array([np.array(Image.open(f)) for f in files])  # (B, n=100*100)
images = np.invert(images).astype(int)  # invert the images to (black=1, white=0)
width, height = images[0].shape
B, n = len(images), width * height  # n = 100x100 = 10,000 units with 10,000x10,000 = 100,000,000 weights


# Tracking
log_id = 'RBM - '
now = datetime.now().strftime('%b%d %H-%M-%S')
train_writer = SummaryWriter(f'runs/{log_id}{now}', flush_secs=2)


# Train the network
images_batch = torch.tensor(images.reshape(B, -1), device=DEVICE, dtype=torch.float)
net = RestrictedBoltzmannMachine(n, n//10, device=DEVICE)
print(f'Fitting all the {B} images..')
pbar = trange(1, EPOCHS+1, desc='EPOCH')
for epoch in pbar:
    k = 1 + epoch//K_EPOCHS
    for X, batch_fraction in batches(images_batch, batch_size=BATCH_SIZE, shuffle=True, device=DEVICE):
        net.update(X, lr=LEARN_RATE, k_reconstructions=k)

    sq_loss = ((images_batch - net.sample(images_batch)) ** 2).mean()
    pbar.set_postfix(lr=LEARN_RATE, k_reconstructions=k, reconstruction_sq_error=f"{sq_loss:.4f}")
    train_writer.add_scalar('a/Reconstruction error (use but don\'t trust)', sq_loss, epoch)
    if epoch == 1 or epoch % 10 == 0:
        train_writer.add_histogram('h/Weights', net.W, epoch)
        train_writer.add_histogram('h/Bias visible', net.v_bias, epoch)
        train_writer.add_histogram('h/Bias hidden', net.h_bias, epoch)


# Reconstruct the images from noisy images
images_noisy = images.copy()
images_noisy[:, 20:80, 20:80] = 0.
images_sampled = []
images_reconstructed = []
n_samples, burn_in = 6, 2
for i, img_noisy in enumerate(images_noisy):
    print(f'reconstructing image {i+1}/{B}')
    img_noisy = torch.tensor(img_noisy.ravel(), device=DEVICE, dtype=torch.float)
    samples = net.sample(img_noisy, burn_in=burn_in, n_samples=n_samples).detach().cpu().numpy().reshape(n_samples, width, height)
    images_sampled.append(samples)
    reconstructed = net.reconstruct(img_noisy).detach().cpu().numpy().reshape(width, height)
    images_reconstructed.append(reconstructed)


# Plot the corrupted and reconstructed images for comparison
fig, ax = plt.subplots(B, 4, figsize=(+n_samples+3, B * 1.3), gridspec_kw={'width_ratios': [2, 2, 2, n_samples]})
for i, (img, img_noisy, img_reconstructed, img_sampled) in enumerate(zip(images, images_noisy, images_reconstructed, images_sampled)):
    ax[i, 0].imshow(img, cmap='gray_r'); ax[i, 0].axis('off')
    ax[i, 1].imshow(img_noisy, cmap='gray_r'); ax[i, 1].axis('off')
    ax[i, 2].imshow(img_reconstructed, cmap='gray_r'); ax[i, 2].axis('off')
    ax[i, 3].imshow(np.hstack(img_sampled), cmap='gray_r'); ax[i, 3].axis('off')
plt.suptitle(f'Pattern  →  Corrupted  →  Reconstructed  →  Sampled({n_samples})')
plt.tight_layout()
plt.show()

