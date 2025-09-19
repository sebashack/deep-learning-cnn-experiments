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
