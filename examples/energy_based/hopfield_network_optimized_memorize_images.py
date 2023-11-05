import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from models.energy_based_models import HopfieldNetwork, HopfieldNetworkOptimized
import os

DEVICE = 'cuda'
DATA_DIR = './data/BW_images/'

# Prepare images (data)
files = [DATA_DIR+f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
images = np.array([np.array(Image.open(f)) for f in files])  # (B, n=100*100)
images = np.invert(images).astype(int)  # invert the images to (black=1, white=0)
images = np.where(images == 0, -1, 1)   # (black=1, white=-1)
width, height = images[0].shape
B, n = len(images), width * height  # n = 100x100 = 10,000 units with 10,000x10,000 = 100,000,000 weights


# Train the network
net = HopfieldNetworkOptimized(n, device=DEVICE)
images_batch = torch.tensor(images.reshape(B, -1), device=DEVICE, dtype=torch.float)
print(f'fitting all the {B} images as a batch')
net.fit(images_batch, epochs=3)


# Reconstruct the images from noisy images
images_noisy = images * np.random.choice([1, -1], size=images.shape, p=[0.9, 0.1])
images_noisy[:, 30:70, 30:70] = -1
images_reconstructed = np.ones_like(images_noisy)
for i, img_noisy in enumerate(images_noisy):
    print(f'reconstructing image {i+1}/{B}')
    img_noisy = torch.tensor(img_noisy.ravel(), device=DEVICE, dtype=torch.float)
    images_reconstructed[i] = net.forward(img_noisy).cpu().numpy().reshape(width, height)


# Plot the corrupted and reconstructed images for comparison
fig, ax = plt.subplots(B, 3, figsize=(4, B * 1.3))
for i, (img, img_noisy, img_reconstructed) in enumerate(zip(images, images_noisy, images_reconstructed)):
    ax[i, 0].imshow(img, cmap='gray_r', aspect='equal'); ax[i, 0].axis('off')
    ax[i, 1].imshow(img_noisy, cmap='gray_r', aspect='equal'); ax[i, 1].axis('off')
    ax[i, 2].imshow(img_reconstructed, cmap='gray_r', aspect='equal'); ax[i, 2].axis('off')
plt.suptitle('Pattern  →  Corrupted  →  Reconstructed')
plt.tight_layout()
plt.show()

