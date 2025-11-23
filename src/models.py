import torch
import torch.nn as nn

# Always default to float32 to avoid NNPack dtype issues
torch.set_default_dtype(torch.float32)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Do NOT pass dtype/device kwargs here; move the whole model later.
        self.f = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.f(x)
        x = torch.flatten(x, 1)
        return self.head(x)

class SmallMLP(nn.Module):
    def __init__(self, in_dim=600, num_classes=100):
        super().__init__()
        # Do NOT pass dtype/device kwargs here; move the whole model later.
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(True),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.net(x)
