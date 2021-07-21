# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, norm):
        super(DoubleConv, self).__init__()
        if norm == "batch":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        elif norm == "":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        elif norm == "instance":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        elif norm == "group":
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(out_ch, out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(out_ch, out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm):
        super(Inconv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, norm)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, norm))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm, bilinear=True):
        super(Up, self).__init__()

        self.conv = DoubleConv(in_ch, out_ch, norm)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode="nearest")
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UpIncBoundary(Up):
    def __init__(self, in_ch, inter_ch, out_ch):
        super().__init__(in_ch + inter_ch, int(in_ch + inter_ch / 2))
        self.up1 = nn.ConvTranspose2d(in_ch, inter_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + inter_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = torch.cat([x, x3], dim=1)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch, sig=True):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.Sigmoid()
        self.sig = sig

    def forward(self, x):
        x = self.conv(x)
        if self.sig:
            x = self.act(x)
        return x


class Outconv2(nn.Module):
    def __init__(self, in_ch, out_ch, sig=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, int(in_ch / 2), 1)
        self.conv2 = nn.Conv2d(int(in_ch / 2), out_ch, 1)
        if sig:
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        x = self.act(x)
        return x
