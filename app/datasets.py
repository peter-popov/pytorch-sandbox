import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

import numpy as np
from einops import rearrange


class SampleDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()

        self.cifar10 = CIFAR10(root='./data', train=train, download=True, transform=transform)
        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, i):
        return self.cifar10[i]