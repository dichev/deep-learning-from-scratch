import torch
from torchvision import datasets
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, random_split


def MNIST(batch_size, train_val_split=(.8, .2), transforms=None):
    if transforms is None:
        transforms = [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),    # normalized to [0,1]
            # T.Normalize((0.1307,), (0.3081,))      # mean and std are computed over the training dataset
        ]
    transforms = T.Compose(transforms)

    train_split, val_split = train_val_split
    train_dataset = datasets.MNIST('./data/', download=True, train=True, transform=transforms)
    test_dataset = datasets.MNIST('./data/', download=True, train=False, transform=transforms)
    if val_split > 0:
        train_dataset, val_dataset = random_split(train_dataset, (train_val_split[0], train_val_split[1]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) if val_split > 0 else None

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    train_loader, val_loader, test_loader = MNIST(batch_size=32)
    X, y = next(iter(train_loader))
    img_grid = make_grid(X, padding=1, pad_value=.5).permute(1, 2, 0)
    plt.imshow(img_grid)
    plt.axis(False)
    plt.title('MNIST')
    plt.show()
