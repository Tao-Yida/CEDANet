import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from torch.nn.modules.utils import _pair


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


class AttentionPool2d(nn.Module):
    """
    Attention-guided pooling with the same interface as nn.AdaptiveAvgPool2d.
    Forward signature matches AdaptiveAvgPool2d: (x) -> y
    Before calling forward, user must set an attention map via `set_attention`:
        att: shape (B, 1, H, W) or (B, H, W)
    The output is:
        out[b, c, i, j] = sum_{h,w in region_ij} x[b,c,h,w] * att[b,1,h,w]
                         --------------------------------------------------
                         sum_{h,w in region_ij} att[b,1,h,w]
    i.e. weighted adaptive average pooling.
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = _pair(output_size)
        self.attention = None

    def set_attention(self, att: torch.Tensor):
        """
        Set the attention map to be used in pooling.
        Args:
            att: Tensor of shape (B, 1, H, W) or (B, H, W)
        """
        if att.dim() == 3:
            att = att.unsqueeze(1)
        if att.dim() != 4 or att.size(1) != 1:
            raise ValueError(f"Attention map must have shape (B,1,H,W), got {tuple(att.shape)}")
        self.attention = att

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: feature map of shape (B, C, H, W)
        Returns:
            Pooled feature of shape (B, C, out_h, out_w)
        """
        if self.attention is None:
            raise RuntimeError("Attention map not set. Call `set_attention` before forward.")
        B, C, H, W = x.shape
        out_h, out_w = self.output_size
        # Ensure attention matches input spatial dims
        att = self.attention
        if att.shape[2:] != (H, W):
            att = F.interpolate(att, size=(H, W), mode="bilinear", align_corners=False)
        # Weighted features
        weighted = x * att
        # Numerator: weighted features downsampled
        num = F.adaptive_avg_pool2d(weighted, (out_h, out_w))
        # Denominator: attention downsampled
        den = F.adaptive_avg_pool2d(att, (out_h, out_w))
        # Avoid division by zero
        out = num / (den + 1e-6)
        return out


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


class LDConv(nn.Module):
    """
    Learnable Deformable Convolution for domain adaptation.
    """

    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride

        # Fix bias parameter type
        use_bias = bias if bias is not None else False
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=use_bias), nn.BatchNorm2d(outc), nn.SiLU()
        )
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.3 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.3 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(torch.arange(row_number, row_number + 1), torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(0, h * self.stride, self.stride), torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + self.p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        x_offset = rearrange(x_offset, "b c h w n -> b c (h n) w")
        return x_offset


# Domain Discriminator with LDConv support
class DomainDiscriminator(nn.Module):
    """
    Domain Discriminator for domain adaptation.
    Supports the option to use LDConv instead of standard convolution.
    """

    def __init__(self, in_channels, num_domains=2, use_ldconv=False, use_attention_pool=False):
        """
        Args:
            in_channels (int): Number of input channels (feature map depth).
            num_domains (int): Number of domains to discriminate (default is 2).
            use_ldconv (bool): Whether to use LDConv instead of regular Conv2d.
            use_attention_pool (bool): Whether to use AttentionPool2d instead of AdaptiveAvgPool2d.
        """
        super().__init__()
        self.use_ldconv = use_ldconv
        self.use_attention_pool = use_attention_pool

        if use_ldconv:
            # Use LDConv to replace standard convolution, num_param=9 approximates 3x3 convolution
            self.conv1 = LDConv(in_channels, in_channels // 2, num_param=9, stride=1)
            self.conv2 = LDConv(in_channels // 2, in_channels // 4, num_param=9, stride=1)
        else:
            # Original convolution implementation
            self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1)

        self.fc = nn.Conv2d(in_channels // 4, num_domains, 1)
        self.relu = nn.ReLU()

        # Pooling layer selection
        if use_attention_pool:
            self.pool = AttentionPool2d(1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feature_map, category_mask=None, grl_lambda=1.0):
        """
        Args:
            feature_map (torch.Tensor): Input feature map of shape (B, C, H, W).
            category_mask (torch.Tensor, optional): Binary mask of shape (B, 1, H, W)
                indicating regions of interest (e.g., smoke vs background).
            grl_lambda (float): Gradient reversal strength for domain adaptation.
        """
        x = grad_reverse(feature_map, grl_lambda)
        if category_mask is not None:
            x = x * category_mask

        x = self.conv1(x)
        if not self.use_ldconv:  # LDConv already includes activation function
            x = self.relu(x)

        x = self.conv2(x)
        if not self.use_ldconv:
            x = self.relu(x)

        # Use attention pooling or normal pooling
        if self.use_attention_pool and category_mask is not None:
            # Set attention map and apply attention pooling
            self.pool.set_attention(category_mask)

        x = self.pool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)


class DomainAdaptiveGenerator(nn.Module):
    """Domain adaptive generator, integrating the original Generator and domain discriminator"""

    def __init__(self, base_generator, feat_channels=32, num_domains=2, domain_loss_weight=0.1, use_ldconv=False, use_attention_pool=False):
        """
        Args:
            base_generator: Original generator model (Generator)
            feat_channels: Number of feature channels for domain discriminator
            num_domains: Number of domains for domain discriminator (usually 2: source and target)
            domain_loss_weight: Weight for domain discrimination loss, to balance generator and discriminator loss
            use_ldconv: Whether to use LDConv in domain discriminator
            use_attention_pool: Whether to use AttentionPool2d in domain discriminator
        """
        super().__init__()
        self.base_generator = base_generator
        self.domain_loss_weight = domain_loss_weight
        self.feat_channels = feat_channels
        self.use_ldconv = use_ldconv
        self.use_attention_pool = use_attention_pool

        # Build domain discriminators for smoke and background regions, supporting LDConv and AttentionPool2d options
        self.domain_disc_smoke = DomainDiscriminator(feat_channels, num_domains, use_ldconv=use_ldconv, use_attention_pool=use_attention_pool)
        self.domain_disc_bg = DomainDiscriminator(feat_channels, num_domains, use_ldconv=use_ldconv, use_attention_pool=use_attention_pool)

    def extract_features(self, x):
        """
        Extract feature maps from input images for domain discrimination
        """
        # Use the ResNet backbone of sal_encoder to extract intermediate features
        with torch.no_grad():
            _, mux, logvarx = self.base_generator.x_encoder(x)
            z_noise = self.base_generator.reparametrize(mux, logvarx)

        # Extract intermediate features from sal_encoder
        features = self._extract_sal_features(x, z_noise)
        return features

    def _extract_sal_features(self, x: torch.Tensor, z):
        """
        Extract intermediate features from sal_encoder
        """
        sal_encoder = self.base_generator.sal_encoder

        # Reconstruct the first half of the forward pass of sal_encoder
        z = torch.unsqueeze(z, 2)
        z = sal_encoder.tile(z, 2, x.shape[sal_encoder.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = sal_encoder.tile(z, 3, x.shape[sal_encoder.spatial_axes[1]])
        x_input = torch.cat((x, z), 1)
        x_input = sal_encoder.conv_depth1(x_input)

        # Through ResNet backbone
        x = sal_encoder.resnet.conv1(x_input)
        x = sal_encoder.resnet.bn1(x)
        x = sal_encoder.resnet.relu(x)
        x = sal_encoder.resnet.maxpool(x)
        x1 = sal_encoder.resnet.layer1(x)
        x2 = sal_encoder.resnet.layer2(x1)
        x3 = sal_encoder.resnet.layer3_1(x2)

        # Use x3 as the feature for domain discrimination and adjust the number of channels
        if x3.size(1) != self.feat_channels:
            if not hasattr(self, "feature_adapter"):
                self.feature_adapter = nn.Conv2d(x3.size(1), self.feat_channels, 1).to(x3.device)
            features = self.feature_adapter(x3)
        else:
            features = x3

        return features

    def forward(self, x, y=None, training=True, lambda_grl=1.0, source_domain=True):
        """
        Args:
            x: Input image
            y: Ground truth label (for training)
            training: Whether in training mode
            lambda_grl: Gradient reversal layer strength
            source_domain: Whether the data is from source domain
        """
        if training:
            base_outputs = self.base_generator(x, y, training=True)
            sal_init_post, sal_ref_post, sal_init_prior, sal_ref_prior, latent_loss, output_post, output_prior = base_outputs

            # Extract features for domain discrimination
            features = self.extract_features(x)

            # Generate class masks for domain discrimination
            seg_prob = torch.sigmoid(sal_ref_post)
            smoke_mask = (seg_prob > 0.5).float()
            bg_mask = 1.0 - smoke_mask

            # Upsample feature maps to match mask size
            if features.size(-1) != smoke_mask.size(-1):
                features_upsampled = F.interpolate(features, size=smoke_mask.shape[-2:], mode="bilinear", align_corners=True)
            else:
                features_upsampled = features

            # Domain discrimination
            d_smoke = self.domain_disc_smoke(features_upsampled, smoke_mask, lambda_grl)
            d_bg = self.domain_disc_bg(features_upsampled, bg_mask, lambda_grl)

            return sal_init_post, sal_ref_post, sal_init_prior, sal_ref_prior, latent_loss, output_post, output_prior, d_smoke, d_bg
        else:
            # Inference mode, only return segmentation result
            return self.base_generator(x, y, training=False)


def compute_domain_loss(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, batch_size):
    """Compute domain discrimination loss"""
    device = d_smoke_src.device

    # Domain labels: source domain=0, target domain=1
    label_src = torch.zeros(batch_size, dtype=torch.long, device=device)
    label_tgt = torch.ones(batch_size, dtype=torch.long, device=device)

    # Cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Source domain loss
    loss_d_smoke_src = criterion(d_smoke_src, label_src)
    loss_d_bg_src = criterion(d_bg_src, label_src)

    # Target domain loss
    loss_d_smoke_tgt = criterion(d_smoke_tgt, label_tgt)
    loss_d_bg_tgt = criterion(d_bg_tgt, label_tgt)

    # Total domain loss
    domain_loss = (loss_d_smoke_src + loss_d_bg_src + loss_d_smoke_tgt + loss_d_bg_tgt) * 0.25

    return domain_loss, {
        "smoke_src": loss_d_smoke_src.item(),
        "bg_src": loss_d_bg_src.item(),
        "smoke_tgt": loss_d_smoke_tgt.item(),
        "bg_tgt": loss_d_bg_tgt.item(),
    }


def compute_domain_accuracy(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt):
    """
    Compute domain discrimination accuracy, used to monitor domain adaptation training effect
    Ideally, the accuracy should be close to 50% (domain confusion successful)
    """
    # Get predictions
    pred_smoke_src = torch.argmax(d_smoke_src, dim=1)
    pred_bg_src = torch.argmax(d_bg_src, dim=1)
    pred_smoke_tgt = torch.argmax(d_smoke_tgt, dim=1)
    pred_bg_tgt = torch.argmax(d_bg_tgt, dim=1)

    # Ground truth labels
    label_src = torch.zeros_like(pred_smoke_src)
    label_tgt = torch.ones_like(pred_smoke_tgt)

    # Compute accuracy
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
    Log domain adaptation training statistics, used to demonstrate that domain adaptation is indeed working
    """
    domain_loss, domain_loss_dict = compute_domain_loss(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, batch_size)
    domain_acc_dict = compute_domain_accuracy(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt)

    # Calculate domain confusion degree (ideal value close to 0.5)
    confusion_score = abs(0.5 - domain_acc_dict["overall"])

    print(f"\nDomain adaptation stats [Epoch {epoch}, Step {step}]:")
    print(f"   Domain discrimination accuracy: {domain_acc_dict['overall']:.3f} (ideal~0.5)")
    print(f"   Domain confusion score: {confusion_score:.3f} (the smaller the better)")
    print(f"   Smoke region: source accuracy={domain_acc_dict['smoke_src']:.3f}, target accuracy={domain_acc_dict['smoke_tgt']:.3f}")
    print(f"   Background region: source accuracy={domain_acc_dict['bg_src']:.3f}, target accuracy={domain_acc_dict['bg_tgt']:.3f}")
    print(f"   Domain loss: total={domain_loss.item():.4f}")

    # Judge domain adaptation effect
    if domain_acc_dict["overall"] > 0.8:
        print("   Domain discriminator is too accurate, may need to increase GRL strength")
    elif domain_acc_dict["overall"] < 0.3:
        print("   Domain discriminator accuracy is too low, may need to decrease GRL strength")
    elif 0.4 <= domain_acc_dict["overall"] <= 0.6:
        print("   Good domain confusion, domain adaptation is working effectively!")

    return {
        "domain_loss": domain_loss.item(),
        "domain_accuracy": domain_acc_dict["overall"],
        "confusion_score": confusion_score,
        "individual_accuracies": domain_acc_dict,
    }


def create_domain_adaptive_model(base_generator, feat_channels=32, num_domains=2, domain_loss_weight=0.1, use_ldconv=False, use_attention_pool=False):
    """
    Create domain adaptive model
    Args:
        base_generator: Base generator
        feat_channels: Number of feature channels
        num_domains: Number of domains
        domain_loss_weight: Domain loss weight
        use_ldconv: Whether to use LDConv (default False, keep backward compatibility)
        use_attention_pool: Whether to use AttentionPool2d (default False, keep backward compatibility)
    """
    model = DomainAdaptiveGenerator(
        base_generator=base_generator,
        feat_channels=feat_channels,
        num_domains=num_domains,
        domain_loss_weight=domain_loss_weight,
        use_ldconv=use_ldconv,
        use_attention_pool=use_attention_pool,
    )
    return model
