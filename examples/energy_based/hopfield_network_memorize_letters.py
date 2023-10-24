import numpy as np
import matplotlib.pyplot as plt
import torch

from networks.energy_based_models import HopfieldNetwork
from data.letters_5x5 import get_patterns

# Data
patterns, _ = get_patterns()
patterns = patterns[:3]  # the not optimized Hopfield network can memorize only 3 patterns
N, width, height = patterns.shape
n = width * height  # 25


# Train the network
net = HopfieldNetwork(n)
X = torch.tensor(patterns.reshape(N, -1), dtype=torch.float)
net.fit(X)


# Visualize the patterns
fig, ax = plt.subplots(N, 3, figsize=(5, N))
for i, pattern in enumerate(patterns):
    pattern = np.array(pattern)
    pattern_noisy = pattern * np.random.choice([1, -1], size=pattern.shape, p=[0.8, 0.2])
    x = torch.tensor(pattern_noisy.ravel(), dtype=torch.float)
    pattern_reconstructed = net.forward(x).reshape(width, height)
    ax[i, 0].imshow(pattern); ax[i, 0].axis('off')
    ax[i, 1].imshow(pattern_noisy); ax[i, 1].axis('off')
    ax[i, 2].imshow(pattern_reconstructed); ax[i, 2].axis('off')
plt.suptitle('Pattern  →  Corrupted  →  Reconstructed')
plt.tight_layout()
plt.show()

