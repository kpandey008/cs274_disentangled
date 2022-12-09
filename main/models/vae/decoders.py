from re import X
from turtle import xcor
import torch
import torch.nn as nn


class MNISTDecoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        self.act = nn.ReLU()

        self.input_mod = nn.Linear(code_size, 8 * 8 * 1024)
        act = nn.ReLU()
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            act,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            act,
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, z):
        x = self.input_mod(z).reshape(-1, 1024, 8, 8)
        x = self.backbone(x)
        return x


class CIFAR10Decoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        self.act = nn.ReLU()

        self.input_mod = nn.Linear(code_size, 8 * 8 * 1024)
        act = nn.ReLU()
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            act,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            act,
            nn.Conv2d(256, 3, 1),
        )

    def forward(self, z):
        x = self.input_mod(z).reshape(-1, 1024, 8, 8)
        x = self.backbone(x)
        return x


class CelebaDecoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        self.act = nn.ReLU()

        self.input_mod = nn.Linear(code_size, 8 * 8 * 1024)
        act = nn.ReLU()
        self.backbone = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            act,
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            act,
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            act,
            nn.Conv2d(128, 3, 1),
        )

    def forward(self, z):
        x = self.input_mod(z).reshape(-1, 1024, 8, 8)
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    model = CIFAR10Decoder(64)
    img = torch.randn(4, 64)
    out = model(img)
