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
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)


def l2_regularisation(m: nn.Module) -> torch.Tensor:
    l2_reg: Optional[torch.Tensor] = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    # If the module has no parameters, return a zero tensor
    if l2_reg is None:
        # Create a scalar zero tensor, ensure it's on the correct device
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
        return torch.mean(torch.stack(c))


class EarlyStopping:
    """Early stopping strategy class"""

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
        """Save best model weights"""
        self.best_weights = model.state_dict().copy()


def calculate_metrics(pred, gt, threshold=0.5):
    """
    Calculate performance metrics
    Args:
        pred: prediction [batch_size, 1, H, W]
        gt: ground truth [batch_size, 1, H, W]
        threshold: binarization threshold
    Returns:
        dict: dictionary containing various metrics
    """
    with torch.no_grad():
        # Binarize prediction
        pred_bin = (torch.sigmoid(pred) > threshold).float()
        gt_bin = gt

        # Flatten all pixels
        pred_flat = pred_bin.view(-1)
        gt_flat = gt_bin.view(-1)

        # Compute confusion matrix
        tp = ((pred_flat == 1) & (gt_flat == 1)).sum().item()
        tn = ((pred_flat == 0) & (gt_flat == 0)).sum().item()
        fp = ((pred_flat == 1) & (gt_flat == 0)).sum().item()
        fn = ((pred_flat == 0) & (gt_flat == 1)).sum().item()

        # Calculate metrics, avoid division by zero
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "iou": iou}


def validate_model(generator, val_loader, device, structure_loss_fn):
    """
    Model validation function (for semi-supervised models)
    Args:
        generator: generator model
        val_loader: validation data loader
        device: computation device
        structure_loss_fn: structure loss function
    Returns:
        tuple: (average loss, metrics dictionary)
    """
    generator.eval()
    val_loss = 0.0
    all_metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    with torch.no_grad():
        for images, gts, trans in val_loader:
            images, gts, trans = images.to(device), gts.to(device), trans.to(device)

            # Forward pass (use only posterior prediction for validation)
            # For semi-supervised models, Generator's forward returns multiple outputs
            pred_post_init, pred_post_ref, pred_prior_init, pred_prior_ref, latent_loss, out_post, out_prior = generator(images, gts)

            # Compute loss
            sal_loss = 0.5 * (structure_loss_fn(pred_post_init, gts) + structure_loss_fn(pred_post_ref, gts))
            val_loss += sal_loss.item()

            # Compute metrics (using initial posterior prediction)
            batch_metrics = calculate_metrics(pred_post_init, gts)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]

    val_loss /= len(val_loader)

    # Compute overall metrics
    total = all_metrics["tp"] + all_metrics["tn"] + all_metrics["fp"] + all_metrics["fn"]
    precision = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fp"] + 1e-8)
    recall = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fn"] + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (all_metrics["tp"] + all_metrics["tn"]) / (total + 1e-8)
    iou = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fp"] + all_metrics["fn"] + 1e-8)

    metrics = {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "iou": iou}

    return val_loss, metrics


def extract_pretrained_model_name(pretrained_path):
    """
    Extract model name from pretrained model path
    Args:
        pretrained_path: pretrained model path
    Returns:
        str: extracted model name
    """
    if pretrained_path is None:
        return None

    # Get filename (remove path)
    filename = os.path.basename(pretrained_path)

    # Remove file extension
    model_name = os.path.splitext(filename)[0]

    # Remove common suffixes (in priority order)
    suffixes_to_remove = ["_no_pretrained_weights", "_pretrained_weights", "_weights", "_checkpoint", "_ckpt"]

    for suffix in suffixes_to_remove:
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
            break

    # Limit length and clean characters
    if len(model_name) > 30:  # Shorten length limit
        model_name = model_name[:30]

    # Replace unsafe characters
    model_name = re.sub(r"[^\w\-_.]", "_", model_name)

    return model_name


def generate_best_model_filename(model_name, pretrained_weights_path=None):
    """
    Generate best model filename
    Args:
        model_name: model name
        pretrained_weights_path: pretrained weights path
    Returns:
        str: generated best model filename
    """
    if pretrained_weights_path is None:
        return f"{model_name}_best_model.pth"
    else:
        pretrained_name = extract_pretrained_model_name(pretrained_weights_path)
        return f"{model_name}_best_from_{pretrained_name}.pth"


def generate_domain_adaptation_model_name(source_dataset_name, target_dataset_name, pretrained_weights_path=None):
    """
    Generate domain adaptation model name based on source and target dataset names and pretrained model
    Args:
        source_dataset_name: source domain dataset name
        target_dataset_name: target domain dataset name
        pretrained_weights_path: pretrained weights path, if None means training from scratch
    Returns:
        str: generated domain adaptation model name
    """
    if pretrained_weights_path is None:
        # No pretrained model used, use source-to-target naming
        return f"{source_dataset_name}_to_{target_dataset_name}_da"
    else:
        # Pretrained model used, need to extract pretrained model name
        pretrained_model_name = extract_pretrained_model_name(pretrained_weights_path)
        return f"{source_dataset_name}_to_{target_dataset_name}_from_{pretrained_model_name}_da"
