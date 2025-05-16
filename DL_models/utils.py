import torch
import numpy as np
import time
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from datetime import datetime


class EarlyStopping:
    """
    早停机制类，用于在验证集性能不再提高时及时停止训练，防止过拟合
    Attributes:
        patience (int): 验证集性能不再提高后等待的轮数
        verbose (bool): 是否打印早停相关信息
        delta (float): 最小的性能改善阈值，低于此值不认为是改善
        counter (int): 当前计数器，记录连续不改善的轮数
        best_score (float): 当前最佳性能
        early_stop (bool): 是否触发早停
        val_loss_min (float): 当前最低的验证损失
        path (str): 最佳模型保存路径
    """

    def __init__(self, patience=7, verbose=True, delta=0, path="checkpoint.pt"):
        """
        初始化早停对象
        Args:
            patience (int): 验证集性能不再提高后等待的轮数
            verbose (bool): 是否打印早停相关信息
            delta (float): 最小的性能改善阈值，低于此值不认为是改善
            path (str): 最佳模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """
        在每轮验证后调用，检查是否触发早停
        Args:
            val_loss (float): 验证集损失
            model (nn.Module): 模型对象，用于保存最佳模型
        """
        score = -val_loss  # 较低的损失对应较高的分数

        # 首次调用，初始化最佳分数
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # 性能未改善 (score < best_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # 性能改善
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        保存模型检查点
        Args:
            val_loss (float): 验证集损失
            model (nn.Module): 模型对象
        """
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        # 保存模型
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def calculate_weights(dataset, num_classes):
    """Calculates class weights based on pixel frequency in the dataset."""
    print("Calculating class weights from training data...")
    start_time = time.time()
    class_pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    total_pixels = 0
    # Use a DataLoader to iterate efficiently, even if batch_size=1
    # Set num_workers > 0 if I/O is the bottleneck, but 0 might be simpler for debugging
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    processed_samples = 0
    for _, masks in loader:  # We only need the masks
        # masks shape is (1, H, W)
        masks_np = masks.numpy()
        unique, counts = np.unique(masks_np, return_counts=True)
        for cls_label, count in zip(unique, counts):
            if 0 <= cls_label < num_classes:  # Ensure label is valid
                class_pixel_counts[cls_label] += count
        total_pixels += masks.numel()
        processed_samples += 1
        if processed_samples % 100 == 0:  # Print progress
            print(f"  Processed {processed_samples}/{len(loader.dataset)} samples for weight calculation...")

    end_time = time.time()
    print(f"Weight calculation finished in {end_time - start_time:.2f} seconds.")
    print(f"Total pixels counted in training set: {total_pixels}")
    print(f"Pixel counts per class (Training Set): {class_pixel_counts}")

    # Handle cases where a class might be entirely absent in the training set
    if (class_pixel_counts == 0).any():
        print("Warning: One or more classes have zero pixels in the training set!")
        # Assign a very small count to avoid division by zero, or handle differently
        class_pixel_counts[class_pixel_counts == 0] = 1

    # Calculate weights (inverse frequency)
    weights = total_pixels / (num_classes * class_pixel_counts)  # Removed epsilon as we handle zero counts now
    # Normalize weights (optional, can help stability, e.g., make smallest weight 1)
    # weights = weights / weights.min()
    print(f"Calculated raw weights: {weights}")
    return weights.float()  # Return as float tensor


def plot_training_curves(
    history,
    save_dir,
    model_name="model",
    timestamp=None,
    extra_title="",
):
    """
    绘制训练过程中的 loss、准确率、召回率、F1 分数等指标，并保存图片
    history: dict，包含 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_recall', 'val_recall', 'train_f1', 'val_f1' 等 key
    save_dir: 图片保存目录
    model_name: 模型名
    timestamp: 时间戳字符串
    extra_title: 附加到图片标题和文件名的内容
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 取所有指标的最短长度，防止长度不一致
    min_len = min(len(v) for v in history.values() if isinstance(v, list) and len(v) > 0)
    if min_len == 0:
        print("No data to plot!")
        return
    plt.figure(figsize=(12, 8))
    epochs = range(1, min_len + 1)

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"][:min_len], label="Train Loss")
    plt.plot(epochs, history["val_loss"][:min_len], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_acc"][:min_len], label="Train Acc")
    plt.plot(epochs, history["val_acc"][:min_len], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    # Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_recall"][:min_len], label="Train Recall")
    plt.plot(epochs, history["val_recall"][:min_len], label="Val Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.title("Recall")

    # F1
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_f1"][:min_len], label="Train F1")
    plt.plot(epochs, history["val_f1"][:min_len], label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("F1 Score")

    plt.tight_layout()
    fname = f"{model_name}_{timestamp}{('_' + extra_title) if extra_title else ''}_curve.png"
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    history = {
        "train_loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.5, 0.4],
        "train_acc": [0.7, 0.8, 0.9],
        "val_acc": [0.6, 0.7, 0.8],
        "train_recall": [0.6, 0.7, 0.8],
        "val_recall": [0.5, 0.6, 0.7],
        "train_f1": [0.65, 0.75, 0.85],
        "val_f1": [0.55, 0.65, 0.75],
    }
    plot_training_curves(history, "./plots", model_name="example_model")
