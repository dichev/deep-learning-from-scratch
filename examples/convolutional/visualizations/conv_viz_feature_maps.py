import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from lib.layers import Linear, Sequential, Conv2d, MaxPool2d
from lib.functions.activations import relu, softmax

size = 360
filters = 3

# Prepare image
image = np.array(Image.open('./data/images/cat-tiger_gray.jpg'))  # 360x480
image = image[:, 60:420]                                  # crop to 360x360
image = (image - image.mean()) / image.std()

with torch.no_grad():
    # Model
    net = Sequential(
        Conv2d(in_channels=1, out_channels=filters, kernel_size=3, padding='same'),
        relu,
        MaxPool2d(kernel_size=2, stride=2),
        Conv2d(in_channels=3, out_channels=filters, kernel_size=3, padding='same'),
    )
    classifier = Sequential(
        lambda x: torch.flatten(x, start_dim=1),
        Linear(input_size=filters*(size//2)**2, output_size=5),
        softmax
    )
    classifier_conv = Sequential(
        Conv2d(in_channels=filters, out_channels=5, kernel_size=size//2),
        lambda x: torch.flatten(x, start_dim=1),
        softmax
    )

    # Forward
    X = torch.tensor(image, dtype=torch.float).reshape(1, 1, size, size)
    out = net.forward(X)
    probs = classifier_conv.forward(out)

    fig, ax = plt.subplots(1, filters+1, figsize=(8, 3))
    fig.suptitle('Filter maps')
    ax[0].imshow(image, cmap='gray')
    ax[0].axis(False)
    for i in range(filters):
        ax[i+1].imshow(out[0, i], cmap='gray')
        ax[i+1].axis(False)
    plt.tight_layout()
    plt.show()
    print(probs)

