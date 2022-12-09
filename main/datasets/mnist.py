import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    def __init__(self, root, transform=None, subsample_size=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")

        if subsample_size is not None:
            assert isinstance(subsample_size, int)

        self.root = root
        self.transform = transform
        self.dataset = MNIST(
            self.root, train=True, download=True, transform=transform, **kwargs
        )
        self.subsample_size = subsample_size

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = np.asarray(img).astype(np.float) / 255.0
        return torch.tensor(img).unsqueeze(0).float()

    def __len__(self):
        return len(self.dataset) if self.subsample_size is None else self.subsample_size


if __name__ == "__main__":
    root = "/home/pandeyk1/datasets/"
    dataset = MNISTDataset(root)
    print(dataset[0].shape)
