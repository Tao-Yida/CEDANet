import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(outputs, targets, num_classes, smooth=1e-6):
    """
    计算多分类 Dice Loss
    outputs: 模型输出 (N, C, H, W) - Logits or Probabilities
    targets: 真实标签 (N, H, W) - LongTensor with class indices
    num_classes: 类别数
    smooth: 防止除以零
    """
    # 将 targets 转换为 one-hot 编码 (N, C, H, W)
    targets_one_hot = (
        F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    )
    # 将 outputs 转换为概率 (如果它们是 logits)
    if outputs.shape[1] == num_classes:  # Check if it has class dimension
        probs = F.softmax(outputs, dim=1)
    else:  # Assume already probabilities if shape doesn't match typical logits output
        probs = outputs

    # 展平
    probs_flat = probs.view(probs.shape[0], num_classes, -1)
    targets_one_hot_flat = targets_one_hot.view(
        targets_one_hot.shape[0], num_classes, -1
    )

    # 计算交集和各自的和
    intersection = torch.sum(probs_flat * targets_one_hot_flat, dim=2)  # (N, C)
    prob_sum = torch.sum(probs_flat, dim=2)  # (N, C)
    target_sum = torch.sum(targets_one_hot_flat, dim=2)  # (N, C)

    # 计算 Dice 系数 per class per batch item
    dice_coefficient = (2.0 * intersection + smooth) / (
        prob_sum + target_sum + smooth
    )  # (N, C)

    # 计算 Dice Loss (通常是 1 - Dice 系数)
    # 对所有类别取平均
    dice_loss_per_class = 1.0 - dice_coefficient  # (N, C)
    # 对类别维度求平均，再对 batch 维度求平均
    return torch.mean(dice_loss_per_class)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, reduction="mean"):
        """
        Tversky Loss for multi-class segmentation.
        alpha: Weight for False Negatives (FN).
        beta: Weight for False Positives (FP).
        smooth: Smoothing factor to prevent division by zero.
        reduction: 'mean', 'sum', or 'none'.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, outputs, targets, num_classes):
        """
        outputs: Logits or probabilities from the model (N, C, H, W).
        targets: Ground truth labels (N, H, W), with values in [0, C-1].
        num_classes: Number of classes.
        """
        # Convert targets to one-hot encoding (N, C, H, W)
        targets_one_hot = (
            F.one_hot(targets.long(), num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Convert outputs to probabilities (if they are logits)
        if outputs.shape[1] == num_classes:
            probs = F.softmax(outputs, dim=1)
        else:
            probs = outputs

        # Flatten
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_one_hot_flat = targets_one_hot.view(
            targets_one_hot.shape[0], num_classes, -1
        )

        # True Positives (TP)
        TP = torch.sum(probs_flat * targets_one_hot_flat, dim=2)
        # False Negatives (FN)
        FN = torch.sum((1 - probs_flat) * targets_one_hot_flat, dim=2)
        # False Positives (FP)
        FP = torch.sum(probs_flat * (1 - targets_one_hot_flat), dim=2)

        # Tversky Index per class per batch item
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )  # (N, C)

        # Tversky Loss (1 - Tversky Index)
        tversky_loss_per_class = 1.0 - tversky_index  # (N, C)

        # Apply reduction
        if self.reduction == "mean":
            # Average over classes and then over batch items
            return torch.mean(tversky_loss_per_class)
        elif self.reduction == "sum":
            return torch.sum(tversky_loss_per_class)
        else:  # 'none'
            return tversky_loss_per_class
