import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np
from typing import List


class Descriptor(nn.Module):
    """
    用于将RGB图像和对应的空间先验图像进行融合，输出单通道特征图
    聚焦于「图像特征」与「先验图」的融合与压缩。
    通过上下采样和多层卷积，使模型在不同尺度上均能处理并融合先验信息，输出目标特征。
    方法简单、高效，常用作 VAE/生成器中的先验编码器或粗预测模块。
    """

    def __init__(self, channel):
        super(Descriptor, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 输入(B, 3, H, W) RGB图像，下采样，扩通道
        self.sconv1 = nn.Conv2d(3, channel, kernel_size=3, stride=2, padding=1)
        self.sconv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.sconv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.sconv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(512)
        # self.bn4 = nn.BatchNorm2d(1024)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 1024)
        # 融合seg先验后的多步下采样支路
        self.conv_pred = nn.Conv2d(1, channel, 3, 1, 1)
        # 融合后降维
        self.conv1 = nn.Conv2d(channel * 2, channel, 3, 2, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel, channel, 3, 2, 1)
        self.conv4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(channel, 1, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)
        # 上采样，2倍放大，与seg先验图像对齐
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        """
        构建一个带多尺度膨胀卷积的分类器模块(Classifier_Module)，用于提取更大感受野的特征
        Args:
            block: Classifier_Module
            dilation_series: 膨胀卷积的膨胀率列表
            padding_series: 膨胀卷积的填充率列表
            NoLabels: 输出通道数
            input_channel: 输入通道数
        Returns:
            block: Classifier_Module
        """
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, input, seg):
        """
        Args:
            input: RGB图像，shape为(B, 3, H, W)
            seg: 先验图像，shape为(B, 1, H, W)
        Returns:
            x: 输出特征图，shape为(B, 1, H/16, W/16)
        """
        x1 = self.sconv1(input)
        # # x1 = self.bn1(x1)
        # x1 = self.relu(x1)
        # x2 = self.sconv2(x1)
        # # x2 = self.bn2(x2)
        # x2 = self.relu(x2)
        # x2 = self.maxpool(x2)
        # x3 = self.sconv3(x2)
        # # x3 = self.bn3(x3)
        # x3 = self.relu(x3)
        # x3 = self.maxpool(x3)
        # x4 = self.sconv4(x3)
        # # x4 = self.bn4(x4)
        # x4 = self.relu(x4)
        # x5 = self.layer5(x4)
        int_feat = self.upsample(x1)
        seg_conv = self.conv_pred(seg)
        # 融合RGB图像和先验图像
        feature_map = torch.cat((int_feat, seg_conv), 1)
        x = self.conv1(feature_map)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        return x


class Encoder_x(nn.Module):
    """
    VAE的先验编码器，只对RGB图像进行编码，输出潜在空间分布
    先验编码器的作用是将输入图像编码为潜在空间分布，用于生成初始显著性图
    """

    def __init__(self, input_channels, channels, latent_size):
        """
        Args:
            input_channels: 输入图像的通道数
            channels: 编码器的通道数
            latent_size: 潜在空间的维度
        """
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        # 5个卷积层，逐步下采样
        # 输出特征图尺寸：feature_map_size = (input_image_size - kernel_size + 2 * padding) / stride + 1
        # 输入通道数：input_channels；输出通道数：channels
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # 输出通道数：2 * channels
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        # 输出通道数：4 * channels
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        # 输出通道数：8 * channels
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        # 输出通道数：8 * channels
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        # 3个全连接层，分别对应不同的输入图像大小
        # 输入图像大小为256时，输出潜在空间分布大小为channels*8*8*8
        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # 输入图像大小为352时，输出潜在空间分布大小为channels*8*11*11
        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # 输入图像大小大于352时，输出潜在空间分布大小为channels*8*14*14
        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        """
        Args:
            input: 输入图像，shape为(B, 3, H, W)
        Returns:
            dist: 潜在空间分布，Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)，为每个维度上独立的正态分布 N(μ,σ)，也叫多维对角高斯
            mu: 潜在空间均值，shape为(B, latent_dim)
            logvar: 潜在空间对数方差，shape为(B, latent_dim)
        """
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))

        # print(output.size())
        if input.shape[2] == 256:
            # print('************************256********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)  # 均值 μ，shape=[B, latent_dim]
            logvar = self.fc2_1(output)  # 对数方差 log σ²
            dist = Independent(
                Normal(loc=mu, scale=torch.exp(logvar)),  # 基础分布：每个维度上独立的正态分布 N(μ,σ)
                1,  # 将最后 1 个维度作为“事件维度”，得到一个多维对角高斯
            )
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif input.shape[2] == 352:
            # print('************************352********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(input.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        # output = output.view(-1, self.channel * 8 * 11 * 11)
        # # print(output.size())
        # # output = self.tanh(output)
        #
        # mu = self.fc1(output)
        # logvar = self.fc2(output)
        # dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # # print(output.size())
        # # output = self.tanh(output)
        #
        # return dist, mu, logvar


class Encoder_xy(nn.Module):
    """
    VAE的后验编码器，接受RGB图像和显著性图，拼接后进行编码，输出潜在空间分布
    """

    def __init__(self, input_channels, channels, latent_size):
        """
        Args:
            input_channels: 输入图像的通道数
            channels: 编码器的通道数
            latent_size: 潜在空间的维度
        """
        super(Encoder_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        # 5个卷积层，逐步下采样
        # 输出特征图尺寸：feature_map_size = (input_image_size - kernel_size + 2 * padding) / stride + 1
        # 输入通道数：input_channels；输出通道数：channels
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # 输出通道数：2 * channels
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        # 输出通道数：4 * channels
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        # 输出通道数：8 * channels
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        # 输出通道数：8 * channels
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        # 3个全连接层，分别对应不同的输入图像大小
        # 输入图像大小为256时，输出潜在空间分布大小为channels*8*8*8
        self.fc1_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        self.fc2_1 = nn.Linear(channels * 8 * 8 * 8, latent_size)
        # 输入图像大小为352时，输出潜在空间分布大小为channels*8*11*11
        self.fc1_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2_2 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        # 输入图像大小大于352时，输出潜在空间分布大小为channels*8*14*14
        self.fc1_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)
        self.fc2_3 = nn.Linear(channels * 8 * 14 * 14, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        """
        Args:
            x: 输入图像，shape为(B, 3, H, W)
        Returns:
            dist: 潜在空间分布，Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)，为每个维度上独立的正态分布 N(μ,σ)，也叫多维对角高斯
            mu: 潜在空间均值，shape为(B, latent_dim)
            logvar: 潜在空间对数方差，shape为(B, latent_dim)
        """
        output = self.leakyrelu(self.bn1(self.layer1(x)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        # print(output.size())
        if x.shape[2] == 256:
            # print('************************256********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 8 * 8)

            mu = self.fc1_1(output)
            logvar = self.fc2_1(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        elif x.shape[2] == 352:
            # print('************************352********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 11 * 11)

            mu = self.fc1_2(output)
            logvar = self.fc2_2(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar
        else:
            # print('************************bigger********************')
            # print(x.size())
            output = output.view(-1, self.channel * 8 * 14 * 14)

            mu = self.fc1_3(output)
            logvar = self.fc2_3(output)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            # print(output.size())
            # output = self.tanh(output)

            return dist, mu, logvar


class Generator(nn.Module):
    """
    实现了基于 VAE 的显著性图生成器(Saliency Generator)，含后验和先验两路编码器，以及一个联合显著性特征解码器。
    训练时同时输出初始/精细显著图及 KL 散度损失，推理时仅输出最终概率图。
    """

    def __init__(self, channel, latent_dim):
        """
        Args:
            channel: 通道数
            latent_dim: 潜在空间维度
        """
        super(Generator, self).__init__()
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)  # 接受RGB图像和潜在空间向量，输出初始/精细显著图
        self.xy_encoder = Encoder_xy(4, channel, latent_dim)  # 接受RGB图像和显著性图，拼接输出后验潜在空间分布
        self.x_encoder = Encoder_x(3, channel, latent_dim)  # 先验编码器，仅编码RGB图像

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        """
        计算后验潜在空间分布与先验潜在空间分布之间的 KL 散度损失
        Args:
            posterior_latent_space: 后验潜在空间分布
            prior_latent_space: 先验潜在空间分布
        Returns:
            kl_div: KL 散度损失(KL(posterior || prior))
        """
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        通过重参数化技巧从潜在空间分布中采样
        Args:
            mu: 潜在空间均值([B, latent_dim])
            logvar: 潜在空间对数方差([B, latent_dim])
        Returns:
            eps: 采样得到的潜在空间向量
        """
        std = logvar.mul(0.5).exp_()  # 标准差([B, latent_dim])，exp_()表示就地操作
        # eps = Variable(std.data.new(std.size()).normal_())
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, y=None, training=True):
        """
        前向传播函数
        Args:
            x: 输入图像([B, 3, H, W]).
            y: 显著性图([B, 1, H, W]).
            training: 是否处于训练模式.
        Returns:
            sal_init_post: 后验初始显著性图([B, 1, H, W]).
            sal_ref_post: 后验精细显著性图([B, 1, H, W]).
            sal_init_prior: 先验初始显著性图([B, 1, H, W]).
            sal_ref_prior: 先验精细显著性图([B, 1, H, W]).
            latent_loss: KL 散度损失的平均值，表示后验潜在空间分布与先验潜在空间分布之间的差异.
            prob_pred: 预测的显著性图([B, 1, H, W])，仅在推理模式下返回.
        过程:
            1. 训练模式下，输入 RGB 图像和显著性图，经过编码器得到后验潜在空间分布和初始/精细显著性图.
            2. 训练模式下，输入 RGB 图像，经过编码器得到先验潜在空间分布和初始/精细显著性图.
            3. 训练模式下，计算后验潜在空间分布与先验潜在空间分布之间的 KL 散度损失.
            4. 推理模式下，输入 RGB 图像，经过编码器得到先验潜在空间分布和预测的显著性图.
        """
        if training:
            # 1. 后验分布
            self.posterior, muxy, logvarxy = self.xy_encoder(torch.cat((x, y), 1))
            # 2. 先验分布
            self.prior, mux, logvarx = self.x_encoder(x)
            # 3. 计算 KL 散度损失
            latent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
            # 4. 重参数化采样
            z_noise_post = self.reparametrize(muxy, logvarxy)
            z_noise_prior = self.reparametrize(mux, logvarx)
            # 5. 通过编码器得到初始/精细显著性图
            self.sal_init_post, self.sal_ref_post = self.sal_encoder(x, z_noise_post)
            self.sal_init_prior, self.sal_ref_prior = self.sal_encoder(x, z_noise_prior)

            self.sal_init_post = F.interpolate(self.sal_init_post, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_ref_post = F.interpolate(self.sal_ref_post, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_init_prior = F.interpolate(self.sal_init_prior, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            self.sal_ref_prior = F.interpolate(self.sal_ref_prior, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=True)
            return self.sal_init_post, self.sal_ref_post, self.sal_init_prior, self.sal_ref_prior, latent_loss
        else:
            # 推理模式，仅输出预测的显著性图
            _, mux, logvarx = self.x_encoder(x)
            z_noise = self.reparametrize(mux, logvarx)
            _, self.prob_pred = self.sal_encoder(x, z_noise)
            return self.prob_pred


class CAM_Module(nn.Module):
    """
    Channel Attention Module (CAM)
    --------------------------------
    实现论文 “Dual Attention Network for Scene Segmentation” 中的通道自注意力。

    参数
    ----
    in_dim : int
        输入特征的通道数 C。

    前向输入
    -------
    x : torch.Tensor
        形状 (B, C, H, W) 的特征图。

    返回
    ----
    out : torch.Tensor
        通道重加权后的特征图，形状同输入。
    attention : torch.Tensor
        (B, C, C) 的通道注意力权重，可选返回。
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.channel_in = in_dim
        # γ 初始为 0，训练过程中学习多少注意力应被叠加
        self.gamma: Parameter = Parameter(torch.zeros(1))  # 会被自动收集到model.parameters()中，在反向传播时会被更新
        # Softmax 在最后一个维度（通道维）做归一化
        self.softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()  # B=batch size, C=通道数, N=H*W
        proj_query = x.view(B, C, -1)  # (B, C, N)，表示每个通道的所有空间位置当作一个向量维度
        proj_key = proj_query.permute(0, 2, 1)  # (B, N, C)，维度变换为了用torch.bmm()计算相似度矩阵

        # 通道相似度矩阵 (B, C, C)
        energy = torch.bmm(proj_query, proj_key)  # energy[b,i,j] 就是第 b 个样本中通道 i 与通道 j 的内积，相当于它们的相似度

        # 稳定训练的 trick：把能量转成“距离”形式
        energy_new = (
            torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        )  # 为每一⾏（即每个通道 j）找出本⾏的最⾼相似度；将相似度 Similarity 变为“距离” Distance
        attention = self.softmax(
            energy_new
        )  # (B, C, C)，大部分通道之间的“距离”会大于 0，Softmax 后权重更倾向于不那么相似的通道，从而降低冗余、突出互补信息

        proj_value = proj_query  # (B, C, N)
        out = torch.bmm(attention, proj_value)  # (B, C, N)
        out = out.view(B, C, H, W)  # 恢复空间维度

        out = self.gamma * out + x  # 残差连接
        return out


class PAM_Module(nn.Module):
    """
    Position Attention Module (PAM)
    --------------------------------
    实现论文 “Dual Attention Network for Scene Segmentation” 中的空间自注意力。

    参数
    ----
    in_dim : int
        输入特征的通道数 C。

    前向输入
    -------
    x : torch.Tensor
        形状 (B, C, H, W) 的特征图。

    返回
    ----
    out : torch.Tensor
        位置重加权后的特征图，形状同输入。
    attention : torch.Tensor
        (B, H*W, H*W) 的空间注意力权重，可选返回。
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.channel_in = in_dim

        # query / key 1×1 卷积降到 C/8 通道，减小计算量
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv: nn.Conv2d = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        # value 保持原始通道数
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma: Parameter = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        N = H * W

        # (B, N, C/8)
        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        # (B, C/8, N)
        proj_key = self.key_conv(x).view(B, -1, N)

        # 空间相似度 (B, N, N)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # 值映射 (B, C, N)
        proj_value = self.value_conv(x).view(B, C, N)

        # 加权求和并恢复形状
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(B, C, H, W)

        out = self.gamma * out + x  # 残差连接
        return out


class Classifier_Module(nn.Module):
    """
    Classifier Module with Multi-scale Dilation Convolutions
    多尺度膨胀卷积分类器模块：通过若干个相同输入/输出通道的卷积层，使用不同的膨胀率和填充率来提取多尺度特征，扩大感受野。
    各分支并行处理输入特征图，最后将所有分支的输出相加得到最终输出，得到多尺度上下文信息的输出。
    """

    def __init__(self, dilation_series, padding_series, NumLabels, input_channel):
        """
        Args:
            dilation_series: 膨胀卷积的膨胀率列表
            padding_series: 膨胀卷积的填充率列表
            NumLabels: 输出通道数
            input_channel: 输入通道数
        """
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()  # type: List[nn.Conv2d]
        # 向卷积层列表中添加多个卷积层，每个卷积层使用不同的膨胀率和填充率
        for d, p in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel, NumLabels, kernel_size=3, stride=1, padding=p, dilation=d, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Args:
            x: 输入特征图，形状为 (B, C, H, W)，B为批大小，C为通道数，H和W为特征图的高度和宽度
        Returns:
            out: 输出特征图，形状为 (B, NumLabels, H, W)"""
        out = self.conv2d_list[0](x)  # 第一个卷积层的输出
        # 对于后续的卷积层，将它们的输出与第一个卷积层的输出相加
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out  # out.shape = (B, NumLabels, H, W)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    """
    Channel Attention Layer (CALayer)
    为输入特征图的每个通道生成一个标量权重，以通道注意力的方式突出有用特征、抑制冗余信息。
    该层通过全局平均池化将特征图压缩为通道描述符，然后通过两层卷积生成通道权重。
    """

    def __init__(self, channel, reduction=16):
        """
        Args:
            channel: 输入特征图的通道数
            reduction: 通道压缩比例，默认为16
        """
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (1, 1)  # 将每个通道的特征图压缩为一个点，得到通道描述符，即通道的全局信息
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        通道重加权
        Args:
            x: 输入特征图，形状为 (B, C, H, W)，B为批大小，C为通道数，H和W为特征图的高度和宽度
        Returns:
            x*y: 重加权后的特征图，形状同输入
        """
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    参考Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    实现了残差通道注意力块，包含两个卷积层和一个通道注意力层。
    输入特征图经过两个卷积层和一个通道注意力层后，输出与输入特征图相加，形成残差连接。
    """

    def __init__(self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        Args:
            n_feat: 输入特征图的通道数
            kernel_size: 卷积核大小，默认为3
            reduction: 通道压缩比例，默认为16
            bias: 是否使用偏置，默认为True
            bn: 是否使用批归一化，默认为False
            act: 激活函数，默认为ReLU
            res_scale: 残差缩放系数，默认为1
        """
        super(RCAB, self).__init__()
        modules_body = []
        # 堆叠两个卷积层和一个通道注意力层（可选批归一化+激活函数）
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:  # 第一个卷积层后添加激活函数
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))  # 在两个卷积层后添加通道注意力层
        self.body = nn.Sequential(*modules_body)  # 将所有层组合成一个序列，*表示解包列表，相当于(nn.Conv2d, nn.BatchNorm2d, nn.ReLU, CALayer)
        self.res_scale = res_scale  # 残差缩放系数，用于控制残差连接的强度

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        """
        默认卷积层，输出特征图的通道数与输入特征图相同，使用padding保持输出尺寸不变。
        Args:
            in_channels: 输入特征图的通道数
            out_channels: 输出特征图的通道数
            kernel_size: 卷积核大小
            bias: 是否使用偏置，默认为True
        Returns:
            nn.Conv2d: 卷积层
        """
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图，形状为 (B, C, H, W)，B为批大小，C为通道数，H和W为特征图的高度和宽度
        Returns:
            res: 残差连接后的特征图，形状同输入
        """
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class BasicConv2d(nn.Module):
    """
    Basic Convolutional Layer with Batch Normalization
    将一个卷积层和一个批归一化层封装在一起，形成一个基本的卷积模块。
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        """
        Args:
            in_planes: 输入特征图的通道数
            out_planes: 输出特征图的通道数
            kernel_size: 卷积核大小
            stride: 步幅，默认为1
            padding: 填充，默认为0
            dilation: 膨胀率，默认为1
        """
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,  # 不使用偏置，因为后面有批归一化层
            ),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图，形状为 (B, C, H, W)，B为批大小，C为通道数，H和W为特征图的高度和宽度
        Returns:
            x: 输出特征图，形状同输入
        """
        x = self.conv_bn(x)
        return x


class Triple_Conv(nn.Module):
    """
    Triple Convolutional Layer
    将三个卷积层封装在一起，形成一个三重卷积模块。
    """

    def __init__(self, in_channel, out_channel):
        """
        Args:
            in_channel: 输入特征图的通道数
            out_channel: 输出特征图的通道数
        """
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),  # 1x1卷积用于通道数变换
            BasicConv2d(out_channel, out_channel, 3, padding=1),  # 3x3卷积用于特征提取
            BasicConv2d(out_channel, out_channel, 3, padding=1),  # 再次3x3卷积用于进一步特征提取
        )

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入特征图，形状为 (B, C, H, W)，B为批大小，C为通道数，H和W为特征图的高度和宽度
        Returns:
            self.reduce(x): 输出特征图，形状同输入"""
        return self.reduce(x)


class Saliency_feat_encoder(nn.Module):
    """
    It merges features from a backbone with latent variables and applies
    attention mechanisms to produce initial and refined saliency maps.
    """

    # resnet based encoder decoder
    def __init__(self, channel, latent_dim):
        """
        Args:
            channel: 通道数
            latent_dim: 潜在空间维度
        """
        super(Saliency_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()  # 基于ResNet构建特征提取主干
        # self.resnet=res2net50_v1b_26w_4s(pretrained=True)
        # self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )  # 上采样，scale_factor=8表示将特征图的尺寸扩大8倍，用于恢复到原始输入图像的尺寸
        self.dropout = nn.Dropout(0.3)
        # 构建两个多尺度膨胀卷积分类器
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 3)  # 将通道数变为1，用于初始显著性预测

        # 下采样后各层统一降到channel通道数，对ResNet的第2/3/4层输出（256/512/1024通道）分别做1x1卷积降维到channel通道数和两个3x3卷积层特征提取
        self.conv2_1 = nn.Conv2d(512, channel, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(1024, channel, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(2048, channel, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        # 预留的多支路特征融合降维卷积
        self.conv_feat = nn.Conv2d(32 * 5, channel, kernel_size=3, padding=1)

        # 4倍上采样和2倍上采样，用于多层次特征融合时的尺寸对齐
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 空间注意力
        self.pam_attention5 = PAM_Module(channel)
        self.pam_attention4 = PAM_Module(channel)
        self.pam_attention3 = PAM_Module(channel)
        self.pam_attention2 = PAM_Module(channel)
        self.pam_attention1 = PAM_Module(channel)
        # 通道注意力
        self.cam_attention4 = CAM_Module(channel)
        self.cam_attention3 = CAM_Module(channel)
        self.cam_attention2 = CAM_Module(channel)

        self.racb_layer = RCAB(channel * 4)  # 残差通道注意力块，输入特征图的通道数为channel * 4

        # 多尺度特征融合卷积层，为ResNet的第4/3/2层输出特征图分别做特征融合，用于精细预测分支
        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        # 残差通道注意力块，输入特征图的通道数为channel * 2/3/4
        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)
        # 三重卷积层，输入特征图的通道数为2 * channel/3 * channel/4 * channel
        self.conv43 = Triple_Conv(2 * channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        self.conv4321 = Triple_Conv(4 * channel, channel)

        self.HA = HA()  # 引入Holistic Attention模块，将初始显著性图与特征图融合

        # 第二路特征融合分支，完整复制一套融合、注意力、融合、分类器，用于细化显著性
        self.conv4_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.pam_attention4_2 = PAM_Module(channel)
        self.pam_attention3_2 = PAM_Module(channel)
        self.pam_attention2_2 = PAM_Module(channel)
        self.cam_attention4_2 = CAM_Module(channel)
        self.cam_attention3_2 = CAM_Module(channel)
        self.cam_attention2_2 = CAM_Module(channel)
        self.racb_43_2 = RCAB(channel * 2)
        self.racb_432_2 = RCAB(channel * 3)
        self.conv43_2 = Triple_Conv(2 * channel, channel)
        self.conv432_2 = Triple_Conv(3 * channel, channel)
        self.conv4321_2 = Triple_Conv(4 * channel, channel)

        self.spatial_axes = [2, 3]  # 用于tile中指定空间维度的索引

        self.conv_depth1 = BasicConv2d(
            3 + latent_dim, 3, kernel_size=3, padding=1
        )  # 将输入的RGB图像和潜在变量z合并后，经过一个3x3卷积层，输出通道数为3，作为ResNet的输入

        self.layer7 = self._make_pred_layer(
            Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4
        )  # 最终多尺度分类器，输出通道数为1，用于生成精细显著性图

        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NumLabels, input_channel):
        """
        创建一个多尺度膨胀卷积分类器模块
        Args:
            block: 分类器模块类
            dilation_series: 膨胀率列表
            padding_series: 填充率列表
            NumLabels: 输出通道数
            input_channel: 输入特征图的通道数
        Returns:
            block: 分类器模块实例
        """
        return block(dilation_series, padding_series, NumLabels, input_channel)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        用于在指定维度上重复张量 a n_tile 次。
        Args:
            a (torch.Tensor): 输入张量
            dim (int): 指定的维度
            n_tile (int): 重复次数
        Returns:
            torch.Tensor: 重复后的张量
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x, z):
        """
        前向传播函数
        Args:
            x: 输入特征图，形状为 (B, C, H, W)，B为批大小，C为通道数，H和W为特征图的高度和宽度
            z: 潜在变量，形状为 (B, latent_dim)
        Returns:
            sal_init: 初始显著性图，形状为 (B, 1, H, W)
            sal_ref: 精细显著性图，形状为 (B, 1, H, W)"""

        z = torch.unsqueeze(z, 2)  # 在H轴前添加一个维度
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])  # 沿H轴重复z，使其与输入特征图x的高度相同
        z = torch.unsqueeze(z, 3)  # 在W轴前添加一个维度
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])  # 沿W轴重复z，使其与输入特征图x的宽度相同，维度为 (B, latent_dim, H, W)
        x = torch.cat((x, z), 1)  # 与RGB图像x在通道维度上拼接，形成一个新的输入特征图，形状为 (B, C + latent_dim, H, W)
        x = self.conv_depth1(x)  # 将拼接后的特征图通过一个3x3卷积层，输出通道数为3，作为ResNet的输入

        # ResNet特征提取
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        # 特征融合和注意力机制
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv2_feat1 = self.pam_attention2(conv2_feat)
        conv2_feat2 = self.cam_attention2(conv2_feat)
        conv2_feat = conv2_feat1 + conv2_feat2
        conv3_feat = self.conv3(x3)
        conv3_feat1 = self.pam_attention3(conv3_feat)
        conv3_feat2 = self.cam_attention3(conv3_feat)
        conv3_feat = conv3_feat1 + conv3_feat2
        conv4_feat = self.conv4(x4)
        conv4_feat1 = self.pam_attention4(conv4_feat)
        conv4_feat2 = self.cam_attention4(conv4_feat)
        conv4_feat = conv4_feat1 + conv4_feat2
        conv4_feat = self.upsample2(conv4_feat)

        # 同时处理conv4_feat和conv3_feat，生成初始显著性图，多层融合
        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)
        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)
        # conv432 = self.conv432(conv432)
        #
        # conv432 = self.upsample2(conv432)
        # conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        # conv4321 = self.racb_4321(conv4321)
        # conv4321 = self.conv4321(conv4321)

        sal_init = self.layer6(conv432)  #  # 初始显著性图，形状为 (B, 1, H, W)

        # 第二路特征融合分支，完整复制一套融合、注意力、融合、分类器，用于细化显著性
        x2_2 = self.HA(sal_init.sigmoid(), x2)  # 用初始显著性指导中层特征
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 16 x 16
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 8 x 8

        conv2_feat = self.conv2_2(x2_2)
        conv2_feat1 = self.pam_attention2_2(conv2_feat)
        conv2_feat2 = self.cam_attention2_2(conv2_feat)
        conv2_feat = conv2_feat1 + conv2_feat2
        conv3_feat = self.conv3_2(x3_2)
        conv3_feat1 = self.pam_attention3_2(conv3_feat)
        conv3_feat2 = self.cam_attention3_2(conv3_feat)
        conv3_feat = conv3_feat1 + conv3_feat2
        conv4_feat = self.conv4_2(x4_2)
        conv4_feat1 = self.pam_attention4_2(conv4_feat)
        conv4_feat2 = self.cam_attention4_2(conv4_feat)
        conv4_feat = conv4_feat1 + conv4_feat2

        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1)
        conv43 = self.racb_43_2(conv43)
        conv43 = self.conv43_2(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432_2(conv432)

        conv432 = self.conv432_2(conv432)

        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321(conv4321)
        sal_ref = self.layer7(conv4321)

        return self.upsample8(sal_init), self.upsample4(sal_ref)

    def initialize_weights(self):
        """
        初始化ResNet的权重
        """
        res50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # 加载预训练的ResNet50模型
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():  # 直接映射：如果自定义 ResNet 中的层名与预训练模型中的层名完全匹配，直接复制权重
                v = pretrained_dict[k]
                all_params[k] = v
            elif "_1" in k:  # 对于包含"_1"的层名，去除"_1"后缀来匹配原始层名
                name = k.split("_1")[0] + k.split("_1")[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif "_2" in k:  # 对于包含"_2"的层名，去除"_2"后缀来匹配原始层名
                name = k.split("_2")[0] + k.split("_2")[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
