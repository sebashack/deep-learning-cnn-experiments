import sys

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler

import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage, RandomHorizontalFlip, Normalize, Compose

# local modules
from data_generation.image_classification import generate_dataset
from dataset.transformed_tensor import TransformedTensorDataset
from model.manager import ModelManager
from model.binary_classification import BinaryClassification


def make_balanced_sampler(y_tensor, seed=None):
    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator()

    classes, counts = y_tensor.unique(return_counts=True)
    class_weights = 1.0 / counts.float()
    sample_weights = class_weights[y_tensor.squeeze().long()]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True,
    )


# NCHW: (number of images, channels, height, width)
def main():
    seed = 13
    rand_gen = torch.manual_seed(seed)

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
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=16, sampler=make_balanced_sampler(y_train_tensor, seed=42)
    )

    # Validation loader
    x_val_tensor = x_tensor[val_indices]
    y_val_tensor = y_tensor[val_indices]

    val_composer = Compose([Normalize(mean=(0.5), std=(0.5))])
    val_dataset = TransformedTensorDataset(x_val_tensor, y_val_tensor, val_composer)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16)

    h = x_val_tensor.shape[2]
    w = x_val_tensor.shape[3]

    # Model configuration
    lr = 0.1
    model = BinaryClassification(in_features=h * w)
    optimizer_logistic = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Manager setup
    manager = ModelManager(model, loss_fn, optimizer, tag="binary_classification")
    manager.set_seed(seed)
    manager.set_loaders(train_loader=train_loader, val_loader=val_loader)
    manager.set_tensorboard()
    manager.add_graph()

    manager.train(n_epochs=100)


if __name__ == "__main__":
    sys.exit(main())
