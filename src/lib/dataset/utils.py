import torch
from torch.utils.data import WeightedRandomSampler


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


def compute_mean_std(loader):
    n_channels = next(iter(loader))[0].shape[1]
    mean = torch.zeros(n_channels)
    sq_sum = torch.zeros(n_channels)
    total_pixels = 0

    for images, _ in loader:
        b, c, h, w = images.shape
        pixels = b * h * w

        mean += images.sum(dim=[0, 2, 3])
        sq_sum += (images**2).sum(dim=[0, 2, 3])

        total_pixels += pixels

    mean /= total_pixels
    std = ((sq_sum / total_pixels) - mean**2).sqrt()

    return mean, std
