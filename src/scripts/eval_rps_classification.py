import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local modules
from data_generation.image_classification import download_rps
from dataset.utils import compute_mean_std
from model.rps import RpsCNN
from model.manager import ModelManager


# NCHW: (number of images, channels, height, width)
def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument(
        "--rundir_name",
        "-n",
        type=str,
        required=True,
        help="Name of run directory",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=int,
        required=True,
        help="Checkpoint index",
    )

    args = parser.parse_args()

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
    test_dataset = ImageFolder(root=data_dir / "rps-test-set", transform=Compose([Resize(28), ToTensor()]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=100)

    ## Model configuration
    lr = 3e-4
    model = RpsCNN(n_features=5, p=0.3)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Manager setup
    seed = 42
    manager = ModelManager(model, loss_fn, optimizer, tag="mnist_classification")
    manager.set_seed(seed)
    ckpt_path = Path(os.getenv("ROOT")) / f"runs/{args.rundir_name}/checkpoints/{args.checkpoint}.ckpt"
    manager.load_checkpoint(ckpt_path)

    ## Eval
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            z = manager.eval(inputs, to_cpu=False)
            probs = F.softmax(z, dim=1)
            predictions = torch.argmax(probs, dim=1)
            predictions = predictions.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)

    avg_precision = precision_score(all_labels, all_predictions, average="macro")
    precision = precision_score(all_labels, all_predictions, average=None)

    avg_recall = recall_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average=None)

    avg_f1 = f1_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average=None)

    print(f"Avg. Precision: {avg_precision:.4f}")
    for i, p in enumerate(precision):
        print(f"  {i}: {p}")

    print(f"Avg. Recall:    {avg_recall:.4f}")
    for i, r in enumerate(recall):
        print(f"  {i}: {r}")

    print(f"Avg. F1:    {avg_f1:.4f}")
    for i, r in enumerate(f1):
        print(f"  {i}: {r}")

    print("--")
    print(f"Accuracy:  {accuracy:.4f}")


if __name__ == "__main__":
    sys.exit(main())
