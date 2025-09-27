import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from torchvision.transforms.v2 import Compose, Normalize, Resize, RandomResizedCrop, CenterCrop, ToImage, ToDtype
from torchvision.datasets import ImageFolder

# local modules
from data_generation.image_classification import download_rps
from model.manager import ModelManager


def main():
    seed = 13
    rand_gen = torch.manual_seed(seed)

    data_dir = Path(os.getenv("DOWNLOAD_DIR")) / "rps-root"
    data_dir.mkdir(parents=True, exist_ok=True)

    download_rps(data_dir)

    ## Data preparation
    # We have to set the mean and std used to train the original model. The following statistic
    # for each RGB channel were computed on the ILSVRC dataset.
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    composer = Compose([Resize(256), CenterCrop(224), ToImage(), ToDtype(torch.float32, scale=True), normalizer])

    # 2528 images CHW: (3, 224, 224) after resizing
    train_dataset = ImageFolder(root=data_dir / "rps", transform=composer)
    # 624 images CHW: (3, 224, 224) after resizing
    val_dataset = ImageFolder(root=data_dir / "rps-test-set", transform=composer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32)

    ## Model preparation
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=3, bias=True)

    # Model configuration
    lr = 3e-4
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Manager setup
    manager = ModelManager(model, loss_fn, optimizer, tag="resnet18_classification")
    manager.set_seed(seed)
    manager.set_loaders(train_loader=train_loader, val_loader=val_loader)
    manager.set_tensorboard()
    manager.add_graph()

    ## Train
    manager.train(n_epochs=10)


if __name__ == "__main__":
    sys.exit(main())
