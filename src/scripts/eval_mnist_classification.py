import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, ToTensor
from torchvision.datasets import MNIST
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local modules
from model.lenet_5 import LeNet5
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

    data_dir = Path(os.getenv("DOWNLOAD_DIR"))
    data_dir.mkdir(parents=True, exist_ok=True)

    ## Data preparation
    composer = Compose([ToTensor(), Normalize(mean=(0.5), std=(0.5))])

    # 10000 images with CHW=(1, 28, 28)
    test_dataset = MNIST(root=data_dir, train=False, download=True, transform=composer)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2)

    ## Model configuration
    lr = 0.1
    model = LeNet5()
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=lr)

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
