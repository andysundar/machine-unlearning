
import os, ssl
from typing import Tuple, Optional
import torch
from torch.utils.data import random_split, TensorDataset
import torchvision
import torchvision.transforms as T
import numpy as np

try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

def get_cifar10(root: str, train: bool, download: bool=False):
    tfm_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    tfm_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform = tfm_train if train else tfm_test
    ds = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    ncls = 10
    im_shape = (3, 32, 32)
    return ds, ncls, im_shape

def get_purchase_synth(root: str, train: bool, n: int = 12000, d: int = 600, ncls: int = 100):
    rng = np.random.default_rng(1337 if train else 1441)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    W = rng.normal(0, 0.25, size=(d, ncls))
    logits = X @ W
    y = logits.argmax(axis=1).astype(np.int64)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    ds = TensorDataset(X, y)
    im_shape = (d,)
    return ds, ncls, im_shape

def split_train_val(dataset, val_fraction: float = 0.1, seed: int = 1337):
    n = len(dataset)
    v = int(round(n * val_fraction))
    t = n - v
    gen = torch.Generator().manual_seed(seed)
    trainset, valset = random_split(dataset, [t, v], generator=gen)
    return trainset, valset
