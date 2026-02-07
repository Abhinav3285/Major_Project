# model/segmentation/unet_model.py
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = nn.functional.interpolate(x, scale_factor=2)
        return self.dec1(x)
