import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.transforms.v2 import Compose, Normalize, Resize, RandomResizedCrop, CenterCrop, ToImage, ToDtype
from torchvision.datasets import ImageFolder

# local modules
from data_generation.image_classification import download_rps
from model.manager import ModelManager


def preprocessed_dataset(model, loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    features = None
    labels = None

    features_list = []
    labels_list = []
    with torch.no_grad():
        model.eval()
        for x, y in loader:
            x = x.to(device)
            output = model(x)

            features_list.append(output.detach().cpu())
            labels_list.append(y.cpu())

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    dataset = TensorDataset(features, labels)

    model.to("cpu")
    return dataset


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
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=AlexNet_Weights.DEFAULT)

    # Freeze all parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier[6] = nn.Identity()

    ## Feature preparation
    # We have set the sixth layer (last one) to the identity one, so the fith layer
    # produces 1-d arrays of size 4096. Our feature datasets will be a tensor of shape
    # (n, 4096) and there is a label per row.
    train_preproc = preprocessed_dataset(model, train_loader)
    val_preproc = preprocessed_dataset(model, val_loader)

    # We could save these feature datasets with:
    # torch.save(train_preproc.tensors, 'rps_preproc.pth')
    # torch.save(val_preproc.tensors, 'rps_val_preproc.pth')

    train_preproc_loader = DataLoader(train_preproc, batch_size=64, shuffle=True)
    val_preproc_loader = DataLoader(val_preproc, batch_size=32)

    ## Model configuration
    # The top model we will train will just replace the layer we removed
    top_model = nn.Sequential(nn.Linear(in_features=4096, out_features=3, bias=True))
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(top_model.parameters(), lr=3e-4)

    ## Manager setup
    manager = ModelManager(top_model, loss_fn, optimizer, tag="alexnet_only_features_classification")
    manager.set_seed(seed)
    manager.set_loaders(train_loader=train_preproc_loader, val_loader=val_preproc_loader)
    manager.set_tensorboard()
    manager.add_graph()

    ## Train
    manager.train(n_epochs=100)

    ## For predictions we would plug our trained top layer to the original model
    # model.classifier[6] = top_model


if __name__ == "__main__":
    sys.exit(main())
