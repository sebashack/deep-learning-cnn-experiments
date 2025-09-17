from torch.utils.data import Dataset


class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self._x = x
        self._y = y
        self._transform = transform

    def __getitem__(self, i):
        x = self._x[i]

        if self._transform:
            x = self._transform(x)

        return x, self._y[i]

    def __len__(self):
        return len(self._x)
