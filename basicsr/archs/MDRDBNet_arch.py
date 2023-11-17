import math
import torch
import torch.nn.functional as F
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class MDRDBNet(nn.Module):
    def __init__(self, in_channels=1, num_feat=64, num_blocks=9, scale=4, num_grow_ch=96,out_channels=1):
        super(MDRDBNet, self).__init__()
        # self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)  # 不需要先提取了
        self.conv_3x3 = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.conv_5x5 = nn.Conv2d(in_channels, num_feat, 5, 1, 2)

        self.body1 = make_layer(ResidualBlock3x3, num_blocks, num_feat=num_feat)
        self.body2 = make_layer(ResidualBlock5x5, num_blocks, num_feat=num_feat, num_ch=num_grow_ch)

        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d( num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)
        self.up = Upsample(scale=scale,num_feat=num_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.f_att = AttentionBlock(64, 64, 64)
        self.edge_extractor = EdgeDetectionLayer()
        self.edge_conv = nn.Conv2d(2, num_feat, 1, 1, 0)  # Adjust the number of output channels to match num_feat

    def forward(self, x):
        feat3x3 = self.conv_3x3(x)
        feat5x5 = self.conv_5x5(x)

        # Apply ResidualDenseBlock to 3x3 and 5x5 features
        x3 = self.body1(feat3x3)
        x5 = self.body2(feat5x5)

        feat = self.f_att(x3,x5)

        # Extract edge features
        edge_feat = self.edge_extractor(x)
        edge_feat = self.edge_conv(edge_feat)

        # Combine features
        # feat = torch.cat((feat, edge_feat), dim=1)  # Concatenate along the channel dimension
        feat = self.f_att(feat,edge_feat)
        # upsample
        feat = self.up(feat)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out



class ResidualBlock3x3(nn.Module):
    def __init__(self, num_feat):
        super(ResidualBlock3x3, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, 2 * num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(3 * num_feat, 2 * num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(5 * num_feat, 2 * num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(7 * num_feat, 2 * num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(9 * num_feat, 64, 3, 1, 1)
        self.se_block = SEBlock(num_feat)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.se_block(x)
        x1_1 = self.act(self.conv1(x))
        x1_2 = self.act(self.conv2(torch.cat((x, x1_1), 1)))
        x1_3 = self.act(self.conv3(torch.cat((x, x1_1, x1_2), 1)))
        x1_4 = self.act(self.conv4(torch.cat((x, x1_1, x1_2, x1_3), 1)))
        x1_5 = self.conv5(torch.cat((x, x1_1, x1_2, x1_3, x1_4), 1))
        return x1_5 + x


class ResidualBlock5x5(nn.Module):
    def __init__(self, num_feat, num_ch):
        super(ResidualBlock5x5, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_ch, 5, 1, 2)
        self.conv2 = nn.Conv2d(num_ch + num_feat, num_ch, 5, 1, 2)
        self.conv3 = nn.Conv2d(2 * num_ch + num_feat, 64, 5, 1, 2)
        self.se_block = SEBlock(num_feat)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.se_block(x)
        x2_1 = self.act(self.conv1(x))
        x2_2 = self.act(self.conv2(torch.cat((x, x2_1), 1)))
        x2_3 = self.conv3(torch.cat((x, x2_1, x2_2), 1))
        return x2_3 + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat, num_ch):
        super(ResidualDenseBlock, self).__init__()
        self.block_3x3 = ResidualBlock3x3(num_feat)
        self.block_5x5 = ResidualBlock5x5(num_feat, num_ch)

    def forward(self, x):
        x3 = self.block_3x3(x)
        x5 = self.block_5x5(x)
        return x3, x5


# 边缘特征提取
class EdgeDetectionLayer(nn.Module):
    def __init__(self):
        super(EdgeDetectionLayer, self).__init__()
        self.edge_filter = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        # Sobel edge detection filters
        sobel_filter = torch.tensor([[[[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]]],
                                     [[[1., 2., 1.],
                                       [0., 0., 0.],
                                       [-1., -2., -1.]]]], dtype=torch.float32)
        self.edge_filter.weight = nn.Parameter(sobel_filter)

    def forward(self, x):
        return self.edge_filter(x)


# 注意力机制
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Conv2d(F_int, 1, 1)

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.psi(F.relu(g + x))
        w = F.softmax(psi, dim=1)
        return w * x + g

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)




