import torch.nn as nn
import torch.nn.functional as F


class RpsCNN(nn.Module):
    def __init__(self, n_features, p=0.0):
        super(RpsCNN, self).__init__()

        self._n_features = n_features
        self._p = p

        # Convolution layers
        self._conv1 = nn.Conv2d(in_channels=3, out_channels=self._n_features, kernel_size=3, bias=True)
        self._conv2 = nn.Conv2d(in_channels=self._n_features, out_channels=self._n_features, kernel_size=3, bias=True)

        # Linear layers
        self._linear1 = nn.Linear(in_features=self._n_features * 5 * 5, out_features=50, bias=True)
        self._linear2 = nn.Linear(in_features=50, out_features=3, bias=True)

        # Dropout layers
        self._drop = nn.Dropout(self._p)

        # Flatten layer
        self._flatten = nn.Flatten()

    def featurizer(self, x):
        """
        Notice that for relu and max pooling we don't need to use the 'nn' layer
        as they don't have parameters and don't learn anything.
        """
        # 3@28x28 -> n_features@26x26
        z = self._conv1(x)
        a = F.relu(z)
        # n_features@26x26 -> n_features@13x13
        a = F.max_pool2d(a, kernel_size=2)

        # n_features@13x13 -> n_features@11x11
        z = self._conv2(a)
        a = F.relu(z)
        # n_features@11x11 -> n_features@5x5
        a = F.max_pool2d(a, kernel_size=2)

        return self._flatten(a)

    def classifier(self, x):
        if self._p > 0.0:
            x = self._drop(x)
        z = self._linear1(x)
        a = F.relu(z)

        if self._p > 0.0:
            a = self._drop(a)
        z = self._linear2(a)

        return z

    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)

        return x
