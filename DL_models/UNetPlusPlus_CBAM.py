# unet_plus_plus_cbam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# 通道注意力：对空间池化，得到每个通道的权重
# 空间注意力：对通道池化，得到每个空间位置的权重


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Args:
        in_channels (int): Number of input channels
        reduction (int): Reduction ratio for the channel attention
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 对输入特征图进行全局平均池化，输出大小为1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 对输入特征图进行全局最大池化，输出大小为1x1
        # 1x1卷积，输入通道数为in_channels，输出通道数为in_channels/reduction，reduction是一个超参数，用于减少通道数

        # 对注意力通道后结果做MLP
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Formula:
            CA(x) = Sigmoid(FC2(ReLU(FC1(AvgPool(x))))) + Sigmoid(FC2(ReLU(FC1(MaxPool(x)))))
        Args:
            x: 输入特征图，形状为 (N, C, H, W)
        Returns:
            out: 通道注意力权重，形状为 (N, C, 1, 1)
        """
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对输入特征图进行全局平均池化，得到 (N, C, 1, 1)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对输入特征图进行全局最大池化，得到 (N, C, 1, 1)
        out = avg_out + max_out  # 将两个通道注意力权重相加，得到 (N, C, 1, 1)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Args:
        kernel_size (int): Size of the convolutional kernel for spatial attention
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        # 卷积核大小为 kernel_size，输入通道为2（平均池化和最大池化拼接），输出通道为1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Formula:
            SA(x) = Sigmoid(Conv2d([AvgPool(x), MaxPool(x)], kernel_size(7)))
        """
        # x: 输入特征图，形状为 (N, C, H, W)
        # 对通道维做平均池化，得到 (N, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 对通道维做最大池化，得到 (N, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 在通道维拼接，得到 (N, 2, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 经过卷积，输出 (N, 1, H, W)
        out = self.conv1(x_cat)
        # 经过sigmoid激活，输出空间注意力权重，形状仍为 (N, 1, H, W)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Args:
        in_channels (int): Number of input channels
        reduction (int): Reduction ratio for the channel attention
        kernel_size (int): Size of the convolutional kernel for spatial attention
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBlockCBAM(nn.Module):
    """ConvBlock followed by CBAM module"""

    def __init__(self, in_channels, out_channels, mid_channels=None, reduction=16, kernel_size=7):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, mid_channels)
        self.cbam = CBAM(out_channels, reduction, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x


# ========= Encoder =========
class UNetPPCBAMEncoder(nn.Module):
    def __init__(self, n_channels=3, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # 使用CBAM增强的卷积块
        self.conv0_0 = ConvBlockCBAM(n_channels, nb_filter[0])
        self.conv1_0 = ConvBlockCBAM(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlockCBAM(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlockCBAM(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlockCBAM(nb_filter[3], nb_filter[4])

    def forward(self, x) -> List[torch.Tensor]:
        x0 = self.conv0_0(x)  # (B,32,H, W)
        x1 = self.conv1_0(self.pool(x0))  # (B,64,H/2,W/2)
        x2 = self.conv2_0(self.pool(x1))  # (B,128,H/4,W/4)
        x3 = self.conv3_0(self.pool(x2))  # (B,256,H/8,W/8)
        x4 = self.conv4_0(self.pool(x3))  # (B,512,H/16,W/16)
        return [x0, x1, x2, x3, x4]  # 列表方便解码器按需取用


# ========= Decoder =========
class UNetPPCBAMDecoder(nn.Module):
    def __init__(self, n_classes=3, deep_supervision=False, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 使用CBAM增强的卷积块进行解码
        self.conv0_1 = ConvBlockCBAM(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlockCBAM(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlockCBAM(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlockCBAM(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlockCBAM(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlockCBAM(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlockCBAM(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlockCBAM(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlockCBAM(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlockCBAM(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, 1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, 1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, 1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, 1)

    def forward(self, feats: List[torch.Tensor]):
        # feats = [x0,x1,x2,x3,x4]
        x0_0, x1_0, x2_0, x3_0, x4_0 = feats

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]
        else:
            return self.final(x0_4)


class UNetPlusPlus_CBAM_Segmentor(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, deep_supervision=False):
        super(UNetPlusPlus_CBAM_Segmentor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        # 使用分离的编码器和解码器架构
        self.encoder = UNetPPCBAMEncoder(n_channels=n_channels, nb_filter=nb_filter)
        self.decoder = UNetPPCBAMDecoder(n_classes=n_classes, deep_supervision=deep_supervision, nb_filter=nb_filter)

    def forward(self, input):
        # 编码器获取特征
        features = self.encoder(input)
        # 解码器生成输出
        return self.decoder(features)
