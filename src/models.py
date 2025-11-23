# src/models.py
import torch
import torch.nn as nn

# Keep default compute in float32 to avoid NNPack dtype issues on CPU
torch.set_default_dtype(torch.float32)

__all__ = ["SmallCNN", "SmallMLP"]

class SmallCNN(nn.Module):
    """
    Compact CNN for CIFAR-10 (≈300k params).
    Feature extractor -> Global avg pool -> Linear head.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # 3x32x32 -> 32x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 128x1x1
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume inputs already on the same device; dtype is enforced in training loop
        x = self.features(x)
        x = torch.flatten(x, 1)  # N x 128
        return self.head(x)


class SmallMLP(nn.Module):
    """
    Simple MLP for tabular-like data (e.g., Purchase-100 synthetic).
    If you pass image-shaped tensors, flatten before calling this model.
    """
    def __init__(self, in_dim: int = 600, num_classes: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.net(x)
