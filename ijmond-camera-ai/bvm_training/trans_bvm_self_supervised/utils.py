import torch.nn as nn
import torch
import numpy as np
from typing import Optional
import torch.nn.functional as F
import os
import re


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip) 


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay


def truncated_normal_(tensor, mean=0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m: nn.Module) -> torch.Tensor:
    l2_reg: Optional[torch.Tensor] = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    # 如果模块没有参数，返回零张量
    if l2_reg is None:
        # 创建一个标量零张量，确保在正确的设备上
        return torch.tensor(0.0, device=next(m.parameters(), torch.tensor(0.0)).device)

    return l2_reg


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        a = len(self.losses)
        b = np.maximum(a - self.num, 0)
        c = self.losses[b:]
        # print(c)
        # d = torch.mean(torch.stack(c))
        # print(d)
        return torch.mean(torch.stack(c))


class EarlyStopping:
    """早停策略类"""

    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        self.best_weights = model.state_dict().copy()


def calculate_metrics(pred, gt, threshold=0.5):
    """
    计算性能指标
    Args:
        pred: 预测结果 [batch_size, 1, H, W]
        gt: 真实标签 [batch_size, 1, H, W]
        threshold: 二值化阈值
    Returns:
        dict: 包含各种指标的字典
    """
    with torch.no_grad():
        # 二值化预测
        pred_bin = (torch.sigmoid(pred) > threshold).float()
        gt_bin = gt

        # 展平所有像素
        pred_flat = pred_bin.view(-1)
        gt_flat = gt_bin.view(-1)

        # 计算混淆矩阵
        tp = ((pred_flat == 1) & (gt_flat == 1)).sum().item()
        tn = ((pred_flat == 0) & (gt_flat == 0)).sum().item()
        fp = ((pred_flat == 1) & (gt_flat == 0)).sum().item()
        fn = ((pred_flat == 0) & (gt_flat == 1)).sum().item()

        # 计算指标，避免除零错误
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "iou": iou}


def validate_model(generator, val_loader, device, structure_loss_fn):
    """
    模型校验函数
    Args:
        generator: 生成器模型
        val_loader: 校验数据加载器
        device: 计算设备
        structure_loss_fn: 结构损失函数
    Returns:
        tuple: (平均损失, 指标字典)
    """
    generator.eval()
    val_loss = 0.0
    all_metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    with torch.no_grad():
        for images, gts, trans in val_loader:
            images, gts, trans = images.to(device), gts.to(device), trans.to(device)

            # 前向传播（仅使用后验预测进行校验）
            pred_post_init, pred_post_ref, _, _, _ = generator(images, gts)

            # 计算损失
            sal_loss = 0.5 * (structure_loss_fn(pred_post_init, gts) + structure_loss_fn(pred_post_ref, gts))
            val_loss += sal_loss.item()

            # 计算指标（使用初始后验预测）
            batch_metrics = calculate_metrics(pred_post_init, gts)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]

    val_loss /= len(val_loader)

    # 计算总体指标
    total = all_metrics["tp"] + all_metrics["tn"] + all_metrics["fp"] + all_metrics["fn"]
    precision = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fp"] + 1e-8)
    recall = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fn"] + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (all_metrics["tp"] + all_metrics["tn"]) / (total + 1e-8)
    iou = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fp"] + all_metrics["fn"] + 1e-8)

    metrics = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "iou": iou}

    return val_loss, metrics


def generate_model_name(dataset_name, pretrained_weights_path=None):
    """
    根据数据集名称和预训练模型生成模型名称
    Args:
        dataset_name: 数据集名称
        pretrained_weights_path: 预训练权重路径，如果为None则表示从头训练
    Returns:
        str: 生成的模型名称
    """
    if pretrained_weights_path is None:
        # 没有使用预训练模型，只使用数据集名称
        return dataset_name
    else:
        # 使用了预训练模型，需要提取预训练模型名称
        pretrained_model_name = extract_pretrained_model_name(pretrained_weights_path)
        return f"{dataset_name}_from_{pretrained_model_name}"


def extract_pretrained_model_name(pretrained_path):
    """
    从预训练模型路径中提取模型名称
    Args:
        pretrained_path: 预训练模型路径
    Returns:
        str: 提取的模型名称
    """
    if pretrained_path is None:
        return None

    # 获取文件名（去掉路径）
    filename = os.path.basename(pretrained_path)

    # 去掉文件扩展名
    model_name = os.path.splitext(filename)[0]

    # 去掉常见的后缀（按优先级排序）
    suffixes_to_remove = ["_no_pretrained_weights", "_pretrained_weights", "_weights", "_checkpoint", "_ckpt"]

    for suffix in suffixes_to_remove:
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
            break

    # 限制长度并清理字符
    if len(model_name) > 30:  # 缩短长度限制
        model_name = model_name[:30]

    # 替换不安全的字符
    model_name = re.sub(r"[^\w\-_.]", "_", model_name)

    return model_name


def generate_checkpoint_filename(epoch, model_name, pretrained_weights_path=None):
    """
    生成检查点文件名
    Args:
        epoch: 当前epoch
        model_name: 模型名称
        pretrained_weights_path: 预训练权重路径
    Returns:
        str: 生成的检查点文件名
    """
    if pretrained_weights_path is None:
        # 从头训练
        return f"{model_name}_epoch_{epoch:03d}_from_scratch.pth"
    else:
        # 使用预训练模型
        pretrained_name = extract_pretrained_model_name(pretrained_weights_path)
        return f"{model_name}_epoch_{epoch:03d}_from_{pretrained_name}.pth"


def generate_best_model_filename(model_name, pretrained_weights_path=None):
    """
    生成最佳模型文件名
    Args:
        model_name: 模型名称
        pretrained_weights_path: 预训练权重路径
    Returns:
        str: 生成的最佳模型文件名
    """
    if pretrained_weights_path is None:
        return f"{model_name}_best_model.pth"
    else:
        pretrained_name = extract_pretrained_model_name(pretrained_weights_path)
        return f"{model_name}_best_from_{pretrained_name}.pth"


# def save_mask_prediction_example(mask, pred, iter):
# 	plt.imshow(pred[0,:,:],cmap='Greys')
# 	plt.savefig('images/'+str(iter)+"_prediction.png")
# 	plt.imshow(mask[0,:,:],cmap='Greys')
#     plt.savefig('images/'+str(iter)+"_mask.png")
