from torch.nn import functional as F
import torch.nn as nn
import torch


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * out_channels, out_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * out_channels, out_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * out_channels, in_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(in_channels, out_channels)
        self.rdb2 = ResidualDenseBlock(in_channels, out_channels)
        self.rdb3 = ResidualDenseBlock(in_channels, out_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_channels, out_channels, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            in_channels = in_channels * 4
        elif scale == 1:
            in_channels = in_channels * 16
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, in_channels=num_feat, out_channels=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            # feat = pixel_unshuffle(x, scale=2)
            pass
        elif self.scale == 1:
            # feat = pixel_unshuffle(x, scale=4)
            pass
        else:
            feat = x
        feat = self.conv_first(feat)
        # 23个RRDB
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def make_layer(basic_block, num_basic_block, **kwarg):

    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]
    return nn.Sequential(*layers)


netG = RRDBNet(3, 3)
X = torch.rand(1, 3, 1280, 720)
output = netG(X)
print(output.shape)
