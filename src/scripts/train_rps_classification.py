import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, Resize
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, CyclicLR, LambdaLR

# local modules
from data_generation.image_classification import download_rps
from dataset.utils import compute_mean_std
from model.manager import ModelManager
from model.rps import RpsCNN


def main():
    seed = 13
    rand_gen = torch.manual_seed(seed)

    data_dir = Path(os.getenv("DOWNLOAD_DIR")) / "rps-root"
    data_dir.mkdir(parents=True, exist_ok=True)

    download_rps(data_dir)

    ## Normalizer preparation
    # 2528 images CHW: (3, 28, 28) after resizing
    tmp_dataset = ImageFolder(root=data_dir / "rps", transform=Compose([Resize(28), ToTensor()]))
    tmp_loader = DataLoader(dataset=tmp_dataset, batch_size=50)

    global_mean, global_std = compute_mean_std(tmp_loader)

    normalizer = Normalize(mean=global_mean, std=global_std)

    ## Data preparation
    composer = Compose([Resize(28), ToTensor(), normalizer])
    # 2528 images CHW: (3, 28, 28) after resizing
    train_dataset = ImageFolder(root=data_dir / "rps", transform=Compose([Resize(28), ToTensor()]))
    # 624 images CHW: (3, 28, 28) after resizing
    val_dataset = ImageFolder(root=data_dir / "rps-test-set", transform=Compose([Resize(28), ToTensor()]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32)

    ## Model configuration
    lr = 3e-4
    model = RpsCNN(n_features=5, p=0.3)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Manager setup
    manager = ModelManager(model, loss_fn, optimizer, tag="rps_classification")
    manager.set_seed(seed)
    manager.set_loaders(train_loader=train_loader, val_loader=val_loader)
    manager.set_tensorboard()
    manager.add_graph()

    manager.train(n_epochs=30)


if __name__ == "__main__":
    sys.exit(main())
