import sys

# import random
import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

# import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler,
# from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, , Resize
# from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, RandomHorizontalFlip, Normalize, Compose

# local modules
from data_generation.image_classification import generate_dataset
from dataset.transformed_tensor import TransformedTensorDataset
from model.manager import ModelManager


def plot_images(images, targets, n_plot=30):
    n_rows = n_plot // 10 + ((n_plot % 10) > 0)
    fig, axes = plt.subplots(n_rows, 10, figsize=(15, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (image, target) in enumerate(zip(images[:n_plot], targets[:n_plot])):
        row, col = i // 10, i % 10
        ax = axes[row, col]
        ax.set_title("#{} - Label:{}".format(i, target), {"size": 12})
        # plot filter channel in grayscale
        ax.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()

    return fig


# fig = plot_images(images, labels, n_plot=30)
# plt.show()


# NCHW: (number of images, channels, height, width)
def main():
    seed = 13
    torch.manual_seed(seed)

    ######################
    ## Data preparation ##
    ######################
    images, labels = generate_dataset(img_size=5, n_images=300, binary=True, seed=seed)

    x_tensor = torch.from_numpy(images // 255).float()
    y_tensor = torch.from_numpy(labels.reshape(-1, 1)).float()
    train_indices, val_indices = random_split(torch.arange(len(x_tensor)), [0.8, 0.2])

    # Training loader
    x_train_tensor = x_tensor[train_indices]
    y_train_tensor = y_tensor[train_indices]

    train_composer = Compose(
        [RandomHorizontalFlip(p=0.5), Normalize(mean=(0.5), std=(0.5))]
    )  # We are normalizing the image matrix to have values -1,1 instead of 0,1.
    train_dataset = TransformedTensorDataset(x_train_tensor, y_train_tensor, train_composer)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

    # Validation loader
    x_val_tensor = x_tensor[val_indices]
    y_val_tensor = y_tensor[val_indices]

    val_composer = Compose([Normalize(mean=(0.5), std=(0.5))])
    val_dataset = TransformedTensorDataset(x_val_tensor, y_val_tensor, val_composer)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16)


if __name__ == "__main__":
    sys.exit(main())
