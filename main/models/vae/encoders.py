import torch
import torch.nn as nn


class MNISTEncoder(nn.Module):
    def __init__(self, code_size, in_ch=1, base_channels=128, channel_mults=[1,2,2,2,2]):
        super().__init__()
        self.code_size = code_size

        self.act = nn.ReLU()

        mod_list = []
        prev_ch = in_ch
        next_ch = channel_mults[0] * base_channels
        for idx in channel_mults:
            mod_list.append(nn.Conv2d(prev_ch, next_ch, 4, stride=2, padding=1))
            mod_list.append(nn.BatchNorm2d(next_ch))
            mod_list.append(self.act)

            prev_ch = next_ch
            next_ch = channel_mults[idx + 1] * next_ch

        self.backbone = nn.Sequential(*mod_list)
        self.mu = nn.Linear(prev_ch, code_size)
        self.logvar = nn.Linear(prev_ch, code_size)

    def forward(self, x):
        b_out = self.backbone(x)
        b_out = torch.flatten(b_out, start_dim=1)
        return self.mu(b_out), self.logvar(b_out)


class CIFAR10Encoder(nn.Module):
    def __init__(self, code_size, in_ch=3, base_channels=128, channel_mults=[1,2,2,2,2]):
        super().__init__()
        self.code_size = code_size

        self.act = nn.ReLU()

        mod_list = []
        prev_ch = in_ch
        next_ch = channel_mults[0] * base_channels
        for idx in channel_mults:
            mod_list.append(nn.Conv2d(prev_ch, next_ch, 4, stride=2, padding=1))
            mod_list.append(nn.BatchNorm2d(next_ch))
            mod_list.append(self.act)

            prev_ch = next_ch
            next_ch = channel_mults[idx + 1] * next_ch

        self.backbone = nn.Sequential(*mod_list)
        self.mu = nn.Linear(prev_ch, code_size)
        self.logvar = nn.Linear(prev_ch, code_size)

    def forward(self, x):
        b_out = self.backbone(x)
        b_out = torch.flatten(b_out, start_dim=1)
        return self.mu(b_out), self.logvar(b_out)


class CelebaEncoder(nn.Module):
    def __init__(self, code_size, in_ch=3, base_channels=128, channel_mults=[1,2,2,2,2]):
        super().__init__()
        self.code_size = code_size

        self.act = nn.ReLU()

        mod_list = []
        prev_ch = in_ch
        next_ch = channel_mults[0] * base_channels
        res = 1
        for idx in channel_mults:
            mod_list.append(nn.Conv2d(prev_ch, next_ch, 4, stride=2, padding=1))
            mod_list.append(nn.BatchNorm2d(next_ch))
            mod_list.append(self.act)

            prev_ch = next_ch
            next_ch = channel_mults[idx + 1] * next_ch
            res *= 2

        self.backbone = nn.Sequential(*mod_list)

        res = 64 ** 2 // res ** 2
        self.mu = nn.Linear(prev_ch * res, code_size)
        self.logvar = nn.Linear(prev_ch * res, code_size)

    def forward(self, x):
        b_out = self.backbone(x)
        b_out = torch.flatten(b_out, start_dim=1)
        return self.mu(b_out), self.logvar(b_out)


if __name__ == '__main__':
    model = CelebaEncoder(64, 3)
    img = torch.randn(4, 3, 64, 64)
    mu, logvar = model(img)
    print(mu.shape)
