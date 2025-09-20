import sys
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, ToTensor
from torchvision.datasets import MNIST

# local modules
from model.manager import ModelManager
from model.lenet_5 import LeNet5


# NCHW: (number of images, channels, height, width)
def main():
    seed = 13
    rand_gen = torch.manual_seed(seed)

    data_dir = Path(os.getenv("DOWNLOAD_DIR"))
    data_dir.mkdir(parents=True, exist_ok=True)
    print(data_dir)

    ######################
    ## Data preparation ##
    ######################
    composer = Compose([ToTensor(), Normalize(mean=(0.5), std=(0.5))])

    # 60000 images with CHW=(1, 28, 28)
    dataset = MNIST(root=data_dir, train=True, download=True, transform=composer)

    train_dataset, val_dataset = random_split(dataset, [0.95, 0.05])

    # Each batch will have shape (64, 1, 28, 28)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64)

    # Model configuration
    lr = 0.1
    model = LeNet5()
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Manager setup
    manager = ModelManager(model, loss_fn, optimizer, tag="mnist_classification")
    manager.set_seed(seed)
    manager.set_loaders(train_loader=train_loader, val_loader=val_loader)
    manager.set_tensorboard()
    manager.add_graph()

    manager.train(n_epochs=50)


if __name__ == "__main__":
    sys.exit(main())
