import torch
from torch import nn
from torch.nn import functional as F


def make_layer(basic_block, num_basic_block, **kwarg):

    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]
    return nn.Sequential(*layers)


class RRDBNet(nn.Module):
    def __init__(self, num_feat):
        super(RRDBNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_feat, num_feat, 3, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_feat, num_feat, 3, 1, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_feat, num_feat, 3, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # x是LG图, 先扩大两倍成HG图尺寸, 再卷积成LG图尺寸, 再扩大两倍成HG图, 然后返回

        x1 = self.net(x)

        output = F.interpolate(x1, scale_factor=2)
        return output + x * 0.2


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_feat=64, num_block=23, out_channels=3):
        super(Generator, self).__init__()

        # 开始卷积层, 将通道数升至num_feat
        self.conv_first = nn.Conv2d(in_channels, num_feat, kernel_size=1)

        # 自制RRDB
        self.net = make_layer(RRDBNet, num_block, num_feat=num_feat)

        # 最后卷积层, 抵消残差0.2的作用
        self.conv_last = nn.Conv2d(num_feat, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        
        x = self.conv_first(x)
        x = self.net(x)
        x = self.conv_last(x)
        return x


