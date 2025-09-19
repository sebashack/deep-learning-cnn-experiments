import torch.nn as nn


class BinaryClassification(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self._flatten = nn.Flatten()
        self._linear0 = nn.Linear(in_features=in_features, out_features=5, bias=True)
        self._relu0 = nn.ReLU()
        self._linear1 = nn.Linear(in_features=5, out_features=3, bias=True)
        self._relu1 = nn.ReLU()
        self._linear2 = nn.Linear(in_features=3, out_features=1, bias=True)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_flat = self._flatten(x)
        z0 = self._linear0(x_flat)
        a0 = self._relu0(z0)

        z1 = self._linear1(a0)
        a1 = self._relu1(z1)

        z2 = self._linear2(a1)

        return self._sigmoid(z2)
