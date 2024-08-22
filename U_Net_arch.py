from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_feat=64, out_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_feat, 3, 1)

        # downsample
        self.conv_down1 = spectral_norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2))
        self.conv_down2 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2))
        self.conv_down3 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2))

        # upsample1
        self.conv_up1 = spectral_norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 2))

        # upsample2
        self.conv_up2 = spectral_norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 2))

        # upsample3
        self.conv_up3 = spectral_norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 2))

        # last
        self.conv_last1 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 2))
        self.conv_last2 = spectral_norm(nn.Conv2d(num_feat, num_feat, 3, 1, 2))
        self.conv_last3 = nn.Conv2d(num_feat, out_channels, 3, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))

        # downsample
        x2 = self.lrelu(self.conv_down1(x1))
        x3 = self.lrelu(self.conv_down2(x2))
        x4 = self.lrelu(self.conv_down3(x3))

        # upsample1
        x5 = F.interpolate(x4, scale_factor=2, mode="nearest")
        x5 = self.lrelu(x5)
        x5 = self.lrelu(self.conv_up1(x5))

        # upsample2
        x6 = x5 + x3
        x6 = F.interpolate(x6, scale_factor=2, mode="nearest")
        x6 = self.lrelu(x6)
        x6 = self.lrelu(self.conv_up2(x6))

        # upsample3
        x7 = x6 + x2
        x7 = F.interpolate(x7, scale_factor=2, mode="nearest")
        x7 = self.lrelu(x7)
        x7 = self.lrelu(self.conv_up3(x7))

        # last
        x8 = x7 + x1
        x8 = self.lrelu(self.conv_last1(x8))
        x8 = self.lrelu(self.conv_last2(x8))
        x8 = self.conv_last3(x8)

        return x8
