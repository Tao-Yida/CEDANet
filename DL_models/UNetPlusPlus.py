# save as: unet_plus_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mid_channels (int, optional): Number of channels in the first convolution. If None, it will be set to out_channels.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels  # if mid_channels is None, set it to out_channels
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


# ========= Encoder =========
class UNetPlusPlusEncoder(nn.Module):
    """
    UNet++ Encoder
    Args:
        n_channels (int): Number of input channels (e.g., 3 for RGB)
        nb_filter (tuple): Number of filters in each layer
    """

    def __init__(self, n_channels=3, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0_0 = ConvBlock(n_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

    def forward(self, x) -> List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            List[torch.Tensor]: List of feature maps from each layer
        """
        x0 = self.conv0_0(x)  # (B,32,H, W)
        x1 = self.conv1_0(self.pool(x0))  # (B,64,H/2,W/2)
        x2 = self.conv2_0(self.pool(x1))  # (B,128,H/4,W/4)
        x3 = self.conv3_0(self.pool(x2))  # (B,256,H/8,W/8)
        x4 = self.conv4_0(self.pool(x3))  # (B,512,H/16,W/16)
        return [x0, x1, x2, x3, x4]  # 列表方便解码器按需取用


# ========= Decoder =========
class UNetPlusPlusDecoder(nn.Module):
    """
    UNet++ Decoder
    Args:
        n_classes (int): Number of output classes
        deep_supervision (bool): Whether to use deep supervision
        nb_filter (tuple): Number of filters in each layer
    """

    def __init__(self, n_classes=3, deep_supervision=False, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, 1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, 1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, 1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, 1)

    def forward(self, feats: List[torch.Tensor]):
        """
        Args:
            feats (List[torch.Tensor]): List of feature maps from the encoder
        Returns:
            torch.Tensor: Output tensor of shape (B, n_classes, H, W)
        """
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


class UNetPlusPlus_Segmentor(nn.Module):
    """
    UNet++ implementation
    Args:
        n_channels (int): Number of input channels (e.g., 3 for RGB)
        n_classes (int): Number of output classes
        deep_supervision (bool): Whether to use deep supervision
    """

    def __init__(self, n_channels=3, n_classes=3, deep_supervision=False):
        super(UNetPlusPlus_Segmentor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]

        # 使用分离的编码器和解码器
        self.encoder = UNetPlusPlusEncoder(n_channels=n_channels, nb_filter=nb_filter)
        self.decoder = UNetPlusPlusDecoder(n_classes=n_classes, deep_supervision=deep_supervision, nb_filter=nb_filter)

    def forward(self, input):
        # 编码器获取特征
        features = self.encoder(input)
        # 解码器生成输出
        return self.decoder(features)


class UNetPlusPlus_Classifier(nn.Module):
    """
    UNet++ 分类器，用于CAM操作
    Args:
        n_classes (int): 输出类别数量
        pretrained (str): 预训练模型路径
        freeze_encoder (bool): 是否冻结编码器参数
        nb_filter (tuple): 每层的滤波器数量
    """

    def __init__(self, n_classes=3, pretrained=None, freeze_encoder=True, nb_filter=(32, 64, 128, 256, 512)):
        super().__init__()
        # 使用与分割模型相同的编码器
        self.encoder = UNetPlusPlusEncoder(n_channels=3, nb_filter=nb_filter)

        # 如果提供预训练模型，加载权重
        if pretrained:
            state_dict = torch.load(pretrained, map_location="cpu")
            # 尝试加载编码器权重，可能需要处理state_dict中的键名
            if "encoder." in list(state_dict.keys())[0]:  # 如果键已经包含'encoder.'前缀
                encoder_state_dict = {
                    k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
                }
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
            else:
                # 如果是分割模型的全部权重，尝试提取编码器部分
                try:
                    self.encoder.load_state_dict(state_dict, strict=False)
                    print("已从预训练模型加载编码器权重")
                except:
                    print("无法加载预训练权重，使用随机初始化")

        # 如果需要冻结编码器
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("编码器参数已冻结")

        # 获取编码器最后一层输出通道数
        encoder_output_channels = nb_filter[-1]  # 默认为512

        # 添加分类头
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(encoder_output_channels, n_classes)

    def forward(self, x):
        # 获取编码器特征
        feats = self.encoder(x)  # feats[-1] => (B,512,H/16,W/16)
        # 全局平均池化并展平
        x = self.gap(feats[-1]).flatten(1) # 只取最后一层特征（存在优化空间）
        # 全连接层分类
        logits = self.fc(x)  # (B,num_classes)
        return logits

    def get_cam_weights(self):
        """返回用于CAM可视化的全连接层权重"""
        return self.fc.weight.data  # shape: (num_classes, enc_out_ch)
