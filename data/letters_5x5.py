import numpy as np
import matplotlib.pyplot as plt

def get_patterns():
    patterns = {
        'A': [
            [-1,  1,  1,  1, -1],
            [ 1, -1, -1, -1,  1],
            [ 1,  1,  1,  1,  1],
            [ 1, -1, -1, -1,  1],
            [ 1, -1, -1, -1,  1]
        ],
        'B': [
            [ 1,  1,  1, -1, -1],
            [ 1, -1, -1,  1, -1],
            [ 1,  1,  1, -1, -1],
            [ 1, -1, -1,  1, -1],
            [ 1,  1,  1, -1, -1]
        ],

        'C': [
            [-1,  1,  1,  1, -1],
            [ 1, -1, -1, -1, -1],
            [ 1, -1, -1, -1, -1],
            [ 1, -1, -1, -1, -1],
            [-1,  1,  1,  1, -1]
        ],

        'D': [
            [ 1,  1,  1, -1, -1],
            [ 1, -1, -1,  1, -1],
            [ 1, -1, -1,  1, -1],
            [ 1, -1, -1,  1, -1],
            [ 1,  1,  1, -1, -1]
        ],

        'E': [
            [ 1,  1,  1,  1,  1],
            [ 1, -1, -1, -1, -1],
            [ 1,  1,  1, -1, -1],
            [ 1, -1, -1, -1, -1],
            [ 1,  1,  1,  1,  1]
        ],

        'F': [
            [ 1,  1,  1,  1, -1],
            [ 1, -1, -1, -1, -1],
            [ 1,  1,  1,  1, -1],
            [ 1, -1, -1, -1, -1],
            [ 1, -1, -1, -1, -1]
        ],
    }
    return np.array(list(patterns.values())), list(patterns.keys())

if __name__ == '__main__':
    patterns, labels = get_patterns()
    N = len(labels)

    plt.figure(figsize=(10, 5))
    for i in range(N):
        plt.subplot(1, N, i + 1)
        plt.imshow(patterns[i], cmap='gray_r')
        plt.title(labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
