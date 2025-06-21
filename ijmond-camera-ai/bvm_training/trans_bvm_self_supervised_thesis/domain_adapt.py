import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Gradient Reversal Layer
def grad_reverse(x, lambda_=1.0):
    """
    Applies a gradient reversal layer to the input tensor.
    Args:
        x (torch.Tensor): Input tensor to apply gradient reversal.
        lambda_ (float): Scaling factor for the gradient reversal.
    Returns:
        torch.Tensor: The input tensor with reversed gradients during backpropagation.
    This layer is used in domain adaptation tasks to encourage the model to learn domain-invariant features.
    by reversing the gradient during backpropagation, effectively penalizing the model for distinguishing between domains
    """
    return GradientReversalLayer.apply(x, lambda_)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) for domain adaptation.
    This layer reverses the gradient during backpropagation,
    effectively allowing the model to learn domain-invariant features.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse the gradient
        return grad_output.neg() * ctx.lambda_, None


# Domain Discriminator, category-aware
class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for domain adaptation.
    This discriminator learns to distinguish between different domains while being aware of the category-specific features in the input.
    """

    def __init__(self, in_channels, num_domains=2):
        """
        Args:
            in_channels (int): Number of input channels (feature map depth).
            num_domains (int): Number of domains to discriminate (default is 2).
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1)
        self.fc = nn.Conv2d(in_channels // 4, num_domains, 1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_map, category_mask=None, grl_lambda=1.0):
        """
        Args:
            feature_map (torch.Tensor): Input feature map of shape (B, C, H, W).
            category_mask (torch.Tensor, optional): Binary mask of shape (B, 1, H, W)
                indicating regions of interest (e.g., smoke vs background).
            grl_lambda (float): Gradient reversal strength for domain adaptation.
        """
        # feature_map: B x C x H x W
        # category_mask: B x 1 x H x W, binary mask indicating category regions (e.g., smoke vs background)
        x = grad_reverse(feature_map, grl_lambda)  # Returns the input with reversed gradients, shape: B x C x H x W
        if category_mask is not None:
            # apply mask to focus on category-specific features: only keep features where category_mask is 1
            x = x * category_mask
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # B x C' x 1 x 1
        x = self.fc(x)  # B x num_domains x 1 x 1
        return x.view(x.size(0), -1)  # B x num_domains


# Usage in the segmentation model:
# Assume `features` is the last feature map before segmentation head,
# and `pred_mask` is the soft segmentation output (B x 1 x H x W).
# We build two discriminators: one for smoke region, one for background.


class DomainAdaptiveGenerator(nn.Module):
    """领域自适应的生成器，集成了原有的Generator和域判别器"""

    def __init__(self, base_generator, feat_channels=32, num_domains=2, domain_loss_weight=0.1):
        """
        Args:
            base_generator: 原有的生成器模型（Generator）
            feat_channels: 特征图的通道数，用于域判别器
            num_domains: 域判别器的域数量（通常为2，表示源域和目标域）
            domain_loss_weight: 域判别损失的权重，用于平衡生成器和域判别器的损失
        """
        super().__init__()
        # 基础生成模型（原有的Generator）
        self.base_generator = base_generator

        # 领域判别器
        self.domain_loss_weight = domain_loss_weight

        # 为烟雾区域和背景区域分别建立域判别器
        self.domain_disc_smoke = DomainDiscriminator(feat_channels, num_domains)
        self.domain_disc_bg = DomainDiscriminator(feat_channels, num_domains)

        # 特征提取器 - 从生成器的编码器部分提取特征
        self.feat_channels = feat_channels

    def extract_features(self, x):
        """
        从输入图像中提取特征图，用于域判别
        Args:
            x: 输入图像，形状为 (B, C, H, W)
        Returns:
            features: 提取的特征图，形状为 (B, feat_channels, H', W')
        这里的特征图是从sal_encoder的ResNet backbone中提取的中间特征，
        用于后续的域判别任务。
        通过使用基础生成器的编码器部分，我们可以获得输入图像的潜在表示，
        然后将其与sal_encoder的特征提取过程结合起来
        """
        # 使用sal_encoder的ResNet backbone提取中间特征
        # 获取x_encoder的潜在表示
        with torch.no_grad():
            _, mux, logvarx = self.base_generator.x_encoder(x)
            z_noise = self.base_generator.reparametrize(mux, logvarx)

        # 从sal_encoder中提取中间特征
        features = self._extract_sal_features(x, z_noise)

        return features

    def _extract_sal_features(self, x: torch.Tensor, z):
        """
        从sal_encoder中提取中间特征
        Args:
            x: 输入图像，形状为 (B, C, H, W)
            z: 从x_encoder中提取的潜在表示，形状为 (B, latent_dim)
        Returns:
            features: 提取的特征图，形状为 (B, feat_channels, H', W')
        """
        # 按照sal_encoder的forward方法提取中间特征
        sal_encoder = self.base_generator.sal_encoder  # 获取sal_encoder实例，类型为Saliency_feat_encoder

        # 重建sal_encoder前向传播的前半部分
        z = torch.unsqueeze(z, 2)
        z = sal_encoder.tile(z, 2, x.shape[sal_encoder.spatial_axes[0]])  # spatial_axes[0]是高度轴
        z = torch.unsqueeze(z, 3)
        z = sal_encoder.tile(z, 3, x.shape[sal_encoder.spatial_axes[1]])  # spatial_axes[1]是宽度轴
        x_input = torch.cat((x, z), 1)
        x_input = sal_encoder.conv_depth1(x_input)  # 经过一个卷积层和一个批归一化层

        # 通过ResNet backbone
        x = sal_encoder.resnet.conv1(x_input)
        x = sal_encoder.resnet.bn1(x)
        x = sal_encoder.resnet.relu(x)
        x = sal_encoder.resnet.maxpool(x)
        x1 = sal_encoder.resnet.layer1(x)  # 256 x 64 x 64
        x2 = sal_encoder.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = sal_encoder.resnet.layer3_1(x2)  # 1024 x 16 x 16

        # 使用x3作为域判别的特征，并调整通道数
        if x3.size(1) != self.feat_channels:
            # 动态创建特征适配器
            if not hasattr(self, "feature_adapter"):
                self.feature_adapter = nn.Conv2d(x3.size(1), self.feat_channels, 1).to(x3.device)
            features = self.feature_adapter(x3)
        else:
            features = x3

        return features

    def forward(self, x, y=None, training=True, lambda_grl=1.0, source_domain=True):
        """
        Args:
            x: 输入图像
            y: 真实标签（训练时）
            training: 是否为训练模式
            lambda_grl: 梯度反转层的强度
            source_domain: 是否为源域数据
        """
        # 使用基础生成器进行前向传播
        if training:
            # 对于目标域，即使提供了标签，我们也使用它们来维持VAE结构
            # 但不会在损失计算中使用目标域的分割监督
            base_outputs = self.base_generator(x, y, training=True)
            # base_outputs: (sal_init_post, sal_ref_post, sal_init_prior, sal_ref_prior, latent_loss, output_post, output_prior)
            sal_init_post, sal_ref_post, sal_init_prior, sal_ref_prior, latent_loss, output_post, output_prior = base_outputs

            # 提取特征进行域判别
            features = self.extract_features(x)

            # 生成类别掩码用于域判别
            # 使用后验预测作为软掩码
            seg_prob = torch.sigmoid(sal_ref_post)

            # 生成烟雾和背景的掩码
            smoke_mask = (seg_prob > 0.5).float()
            bg_mask = 1.0 - smoke_mask

            # 对特征图进行上采样以匹配掩码尺寸
            if features.size(-1) != smoke_mask.size(-1):
                features_upsampled = F.interpolate(features, size=smoke_mask.shape[-2:], mode="bilinear", align_corners=True)
            else:
                features_upsampled = features

            # 域判别
            d_smoke = self.domain_disc_smoke(features_upsampled, smoke_mask, lambda_grl)
            d_bg = self.domain_disc_bg(features_upsampled, bg_mask, lambda_grl)

            return sal_init_post, sal_ref_post, sal_init_prior, sal_ref_prior, latent_loss, output_post, output_prior, d_smoke, d_bg
        else:
            # 推理模式，只返回分割结果
            return self.base_generator(x, y, training=False)


def compute_domain_loss(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, batch_size):
    """计算域判别损失"""
    device = d_smoke_src.device

    # 域标签：源域=0，目标域=1
    label_src = torch.zeros(batch_size, dtype=torch.long, device=device)
    label_tgt = torch.ones(batch_size, dtype=torch.long, device=device)

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 源域损失
    loss_d_smoke_src = criterion(d_smoke_src, label_src)
    loss_d_bg_src = criterion(d_bg_src, label_src)

    # 目标域损失
    loss_d_smoke_tgt = criterion(d_smoke_tgt, label_tgt)
    loss_d_bg_tgt = criterion(d_bg_tgt, label_tgt)

    # 总域损失
    domain_loss = (loss_d_smoke_src + loss_d_bg_src + loss_d_smoke_tgt + loss_d_bg_tgt) * 0.25

    return domain_loss, {
        "smoke_src": loss_d_smoke_src.item(),
        "bg_src": loss_d_bg_src.item(),
        "smoke_tgt": loss_d_smoke_tgt.item(),
        "bg_tgt": loss_d_bg_tgt.item(),
    }


# 添加域适应准确率计算函数，用于验证域判别器是否有效工作
def compute_domain_accuracy(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt):
    """
    计算域判别准确率，用于监控域适应训练效果
    理想情况下，准确率应该接近50%（域混淆成功）
    """
    # 源域标签：0
    # 目标域标签：1

    # 获取预测结果
    pred_smoke_src = torch.argmax(d_smoke_src, dim=1)
    pred_bg_src = torch.argmax(d_bg_src, dim=1)
    pred_smoke_tgt = torch.argmax(d_smoke_tgt, dim=1)
    pred_bg_tgt = torch.argmax(d_bg_tgt, dim=1)

    # 真实标签
    label_src = torch.zeros_like(pred_smoke_src)
    label_tgt = torch.ones_like(pred_smoke_tgt)

    # 计算准确率
    acc_smoke_src = (pred_smoke_src == label_src).float().mean()
    acc_bg_src = (pred_bg_src == label_src).float().mean()
    acc_smoke_tgt = (pred_smoke_tgt == label_tgt).float().mean()
    acc_bg_tgt = (pred_bg_tgt == label_tgt).float().mean()

    overall_acc = (acc_smoke_src + acc_bg_src + acc_smoke_tgt + acc_bg_tgt) / 4

    return {
        "overall": overall_acc.item(),
        "smoke_src": acc_smoke_src.item(),
        "bg_src": acc_bg_src.item(),
        "smoke_tgt": acc_smoke_tgt.item(),
        "bg_tgt": acc_bg_tgt.item(),
    }


def log_domain_adaptation_stats(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, batch_size, epoch, step):
    """
    记录域适应训练统计信息，用于证明域适应确实在工作
    """
    domain_loss, domain_loss_dict = compute_domain_loss(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, batch_size)
    domain_acc_dict = compute_domain_accuracy(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt)

    # 计算域混淆程度 (理想值接近0.5)
    confusion_score = abs(0.5 - domain_acc_dict["overall"])

    print(f"\n 域适应统计 [Epoch {epoch}, Step {step}]:")
    print(f"   域判别准确率: {domain_acc_dict['overall']:.3f} (理想值~0.5)")
    print(f"   域混淆得分: {confusion_score:.3f} (越小越好)")
    print(f"   烟雾区域: 源域准确率={domain_acc_dict['smoke_src']:.3f}, 目标域准确率={domain_acc_dict['smoke_tgt']:.3f}")
    print(f"   背景区域: 源域准确率={domain_acc_dict['bg_src']:.3f}, 目标域准确率={domain_acc_dict['bg_tgt']:.3f}")
    print(f"   域损失: 总计={domain_loss.item():.4f}")

    # 判断域适应效果
    if domain_acc_dict["overall"] > 0.8:
        print("   域判别器过于准确，可能需要增加GRL强度")
    elif domain_acc_dict["overall"] < 0.3:
        print("   域判别器准确率过低，可能需要减少GRL强度")
    elif 0.4 <= domain_acc_dict["overall"] <= 0.6:
        print("   域混淆效果良好，域适应正在有效工作！")

    return {
        "domain_loss": domain_loss.item(),
        "domain_accuracy": domain_acc_dict["overall"],
        "confusion_score": confusion_score,
        "individual_accuracies": domain_acc_dict,
    }


# 领域自适应模型创建函数
def create_domain_adaptive_model(base_generator, feat_channels=32, num_domains=2, domain_loss_weight=0.1):
    """创建领域自适应模型"""
    model = DomainAdaptiveGenerator(
        base_generator=base_generator, feat_channels=feat_channels, num_domains=num_domains, domain_loss_weight=domain_loss_weight
    )
    return model
