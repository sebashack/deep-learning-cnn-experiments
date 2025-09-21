import torch.nn as nn


# For 28×28 MNIST
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN block 1
        self._conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=True)
        self._afn1 = nn.ReLU()
        self._pool1 = nn.MaxPool2d(kernel_size=2)
        # CNN block 2
        self._conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=True)
        self._afn2 = nn.ReLU()
        self._pool2 = nn.MaxPool2d(kernel_size=2)
        # CNN block 3
        self._conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, bias=True)
        self._afn3 = nn.ReLU()
        # Flattening and fully connected layer
        self._flatten = nn.Flatten()
        self._linear1 = nn.Linear(in_features=120, out_features=84, bias=True)
        self._afn4 = nn.ReLU()
        # Output layer
        self._linear2 = nn.Linear(in_features=84, out_features=10, bias=True)

    def forward(self, x):
        z1 = self._conv1(x)
        a1 = self._afn1(z1)
        p1 = self._pool1(a1)
        #
        z2 = self._conv2(p1)
        a2 = self._afn2(z2)
        p2 = self._pool2(a2)
        #
        z3 = self._conv3(p2)
        a3 = self._afn3(z3)
        #
        a3_flat = self._flatten(a3)
        z4 = self._linear1(a3_flat)
        a4 = self._afn4(z4)
        z5 = self._linear2(a4)

        return z5
