from .network_parts import *
import torch.nn as nn
import matplotlib.pyplot as plt


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True, norm="batch"):
        super(UNet, self).__init__()
        self.inc = Inconv(n_channels, 64, norm)
        self.down1 = Down(64, 128, norm)
        self.down2 = Down(128, 256, norm)
        self.down3 = Down(256, 512, norm)
        self.down4 = Down(512, 512, norm)
        self.up1 = Up(1024, 256, norm)
        self.up2 = Up(512, 128, norm)
        self.up3 = Up(256, 64, norm)
        self.up4 = Up(128, 64, norm)
        self.outc = Outconv(64, n_classes, sig)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def forward2(self, x, save_path):
        fig, axs = plt.subplots(1, 11, figsize=(11, 1))
        axs[0].imshow(x[0].sum(dim=0).detach().cpu()[:200, :200], cmap='gray'), axs[0].set_xticks([]), axs[
            0].set_yticks([])
        # plt.savefig(str(save_path.joinpath("00.png")))
        x1 = self.inc(x)
        axs[1].imshow(x1[0].sum(dim=0).detach().cpu()[:200, :200], cmap='seismic'), axs[1].set_xticks([]), axs[
            1].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("01.png")))
        x2 = self.down1(x1)
        axs[2].imshow(x2[0].sum(dim=0).detach().cpu()[:100, :100], cmap='seismic'), axs[2].set_xticks([]), axs[
            2].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("02.png")))
        x3 = self.down2(x2)
        axs[3].imshow(x3[0].sum(dim=0).detach().cpu()[:50, :50], cmap='seismic'), axs[3].set_xticks([]), axs[
            3].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("03.png")))
        x4 = self.down3(x3)
        axs[4].imshow(x4[0].sum(dim=0).detach().cpu()[:25, :25], cmap='seismic'), axs[4].set_xticks([]), axs[
            4].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("04.png")))
        x5 = self.down4(x4)
        axs[5].imshow(x5[0].sum(dim=0).detach().cpu()[:12, :12], cmap='seismic'), axs[5].set_xticks([]), axs[
            5].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("05.png")))
        x = self.up1(x5, x4)
        axs[6].imshow(x[0].sum(dim=0).detach().cpu()[:25, :25], cmap='seismic'), axs[6].set_xticks([]), axs[
            6].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("06.png")))
        x = self.up2(x, x3)
        axs[7].imshow(x[0].sum(dim=0).detach().cpu()[:50, :50], cmap='seismic'), axs[7].set_xticks([]), axs[
            7].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("07.png")))
        x = self.up3(x, x2)
        axs[8].imshow(x[0].sum(dim=0).detach().cpu()[:100, :100], cmap='seismic'), axs[8].set_xticks([]), axs[
            8].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("08.png")))
        x = self.up4(x, x1)
        axs[9].imshow(x[0].sum(dim=0).detach().cpu()[:200, :200], cmap='seismic'), axs[9].set_xticks([]), axs[
            9].set_yticks([])
        # []), plt.tight_layout(), plt.savefig(str(save_path.joinpath("09.png")))
        x = self.outc(x)
        axs[10].imshow(x[0].sum(dim=0).detach().cpu()[:200, :200], cmap='seismic'), axs[10].set_xticks([]), axs[
            10].set_yticks([])
        # []), plt.tight_layout(),
        plt.savefig(str(save_path.joinpath("image.png")), dpi=1000)
        return x


class UNetSmall(nn.Module):
    def __init__(self, n_channels, n_classes, sig=True):
        super().__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Outconv(64, n_classes, sig)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
