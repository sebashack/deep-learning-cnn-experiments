import datetime
import os
import sys
from pathlib import Path

from tqdm import trange
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter


class ModelManager(object):
    def __init__(self, model, loss_fn, optimizer, tag=None):
        prefix = ""
        if tag:
            prefix = f"{tag}_"

        suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self._workdir_path = Path(os.getenv("RUNS_DIR")) / (prefix + f"run_{suffix}")
        self._checkpoints_path = self._workdir_path / "checkpoints"
        self._best_val_loss = sys.float_info.max

        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        print("model is using device:", self._device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("model trainable parameters:", total_params)

        self._train_loader = None
        self._val_loader = None
        self._tb_writer = None

        self._losses = []
        self._val_losses = []
        self._total_epochs = 0

        self._train_step_fn = self._make_train_step_fn()
        self._val_step_fn = self._make_val_step_fn()

        self._scheduler = None
        self._scheduler_needs_val_loss = False

    def set_lr_scheduler(self, scheduler):
        if scheduler.optimizer != self._optimizer:
            raise ValueError("Manager's optimizer does not match scheduler's")

        self._scheduler = scheduler
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self._scheduler_needs_val_loss = True

    def to(self, device):
        try:
            self._device = device
            self._model.to(self._device)
        except RuntimeError:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"couldn't send it to {device}, sending it to {self._device} instead.")
            self._model.to(self._device)

    def set_loaders(self, train_loader, val_loader=None):
        self._train_loader = train_loader
        self._val_loader = val_loader

    def set_tensorboard(self):
        self._tb_writer = SummaryWriter(self._workdir_path)

    def set_seed(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=None):
        self._checkpoints_path.mkdir(parents=True, exist_ok=True)

        if seed:
            self.set_seed(seed)

        pbar = trange(n_epochs, desc="Training", unit="epoch")
        for epoch in pbar:
            self._total_epochs += 1

            loss = self._mini_batch(validation=False)
            self._losses.append(loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                if val_loss:
                    self._scheduler_step(val_loss)
                    self._val_losses.append(val_loss)
                    self._save_checkpoint(epoch, val_loss)

            if self._tb_writer:
                tag_scalar_dict = {"training": loss}
                if val_loss:
                    tag_scalar_dict = {"training": loss, "validation": val_loss}

                self._tb_writer.add_scalars(main_tag="loss", tag_scalar_dict=tag_scalar_dict, global_step=epoch)

            if val_loss is not None:
                pbar.set_postfix({"train_loss": f"{loss:.4f}", "val_loss": f"{val_loss:.4f}"})
            else:
                pbar.set_postfix({"train_loss": f"{loss:.4f}"})

        if self._tb_writer:
            self._tb_writer.close()

    def load_checkpoint(self, ckpt_path: Path):
        checkpoint = torch.load(ckpt_path, weights_only=False)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self._scheduler:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._total_epochs = checkpoint["epoch"]
        self._losses = checkpoint["loss"]
        self._val_losses = checkpoint["val_loss"]
        self._best_val_loss = min(self._val_losses)

        self._model.train()

    def eval(self, x, to_cpu=True):
        self._model.eval()

        x_tensor = torch.as_tensor(x).float()

        with torch.no_grad():
            y_hat_tensor = self._model(x_tensor.to(self._device))

        self._model.train()

        if to_cpu:
            return y_hat_tensor.detach().cpu()
        else:
            return y_hat_tensor.detach()

    def add_graph(self):
        self._checkpoints_path.mkdir(parents=True, exist_ok=True)

        if self._tb_writer:
            x_sample, y_sample = next(iter(self._train_loader))
            self._tb_writer.add_graph(self._model, input_to_model=x_sample.to(self._device))

    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self._model.train()  # Set model in training mode

            yhat = self._model(x)  # forward pass
            loss = self._loss_fn(yhat, y)  # Computes loss
            loss.backward()  # Computes gradients

            self._optimizer.step()  # Update parameters using gradients and the learning rate
            self._optimizer.zero_grad()

            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self._model.eval()  # Set model in evaluation mode
            yhat = self._model(x)
            loss = self._loss_fn(yhat, y)

            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        data_loader = None
        step_fn = None
        if validation:
            data_loader = self._val_loader
            step_fn = self._val_step_fn
        else:
            data_loader = self._train_loader
            step_fn = self._train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def _scheduler_step(self, val_loss):
        if self._scheduler_needs_val_loss:
            self._scheduler.step(val_loss)
        else:
            self._scheduler.step()

    def _save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler else None,
            "loss": self._losses,
            "val_loss": self._val_losses,
        }

        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            (self._workdir_path / "best_checkpoint").write_text(f"{epoch}")
            torch.save(checkpoint, self._checkpoints_path / f"{epoch}.ckpt")
