#!/usr/bin/env python3
# 基于UNet++的图像级分类器训练脚本（优化版）
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 从本地文件导入
from dataset import SmokeDataset_Img
from UNetPlusPlus import UNetPlusPlus_Classifier
from utils import EarlyStopping
import config

# 使用config中的超参数和路径
DEVICE = config.DEVICE
IMG_HEIGHT = config.IMG_HEIGHT
IMG_WIDTH = config.IMG_WIDTH
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
NUM_EPOCHS = config.NUM_EPOCHS
VALIDATION_SPLIT = config.VALIDATION_SPLIT
MODELS_DIR = config.MODELS_DIR
RESULTS_DIR = config.RESULTS_DIR
CLASS_NAMES = config.CLASS_NAMES
NUM_CLASSES = config.NUM_CLASSES


def get_dataloaders(img_dir, label_path, class_map_path, img_height, img_width, batch_size, val_split):
    """创建图像分类数据加载器"""
    # 对图像的变换
    img_transform = T.Compose(
        [
            T.Resize((img_height, img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = img_transform  # 使用相同的变换

    try:
        full_dataset = SmokeDataset_Img(
            img_dir=img_dir,
            label_path=label_path,
            class_map_path=class_map_path,
            transform=img_transform,
        )
    except RuntimeError as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # 划分训练集和验证集
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    # 数据集太小的情况处理
    if n_train == 0 or n_val == 0:
        print(f"Warning: Dataset too small ({len(full_dataset)}) for validation split.")
        train_dataset = full_dataset
        val_dataset = None
    else:
        # 分割数据集
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

        # 为验证集设置transform
        val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 验证集加载器
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def build_model_and_optimizer(pretrained_encoder=None, freeze_encoder=True):
    """构建分类器模型和优化器"""
    model = UNetPlusPlus_Classifier(
        n_classes=NUM_CLASSES, pretrained=pretrained_encoder, freeze_encoder=freeze_encoder
    ).to(DEVICE)

    # 使用BCEWithLogitsLoss处理多标签分类
    criterion = nn.BCEWithLogitsLoss()

    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    return model, criterion, optimizer


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    epoch_train_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels, _ in loader:  # 忽略文件名
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播和损失计算
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累加损失
        epoch_train_loss += loss.item()

        # 收集预测和标签用于计算精度
        preds = torch.sigmoid(outputs) > 0.5  # 二值化
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # 计算平均损失和精度
    avg_train_loss = epoch_train_loss / len(loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())

    print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_train_loss, accuracy


def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in loader:  # 忽略文件名
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播和损失计算
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 收集预测和标签用于计算精度
            preds = torch.sigmoid(outputs) > 0.5  # 二值化
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 计算平均损失和精度
    avg_val_loss = val_loss / len(loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())

    print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_val_loss, accuracy


def plot_training_curves(history, save_dir, model_name):
    """绘制训练曲线并保存"""
    plt.figure(figsize=(10, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # 精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    if "val_accuracy" in history and history["val_accuracy"]:
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存图表
    save_path = os.path.join(save_dir, f"{model_name}_curve.png")
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train UNet++ Classifier")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained encoder weights",
    )
    parser.add_argument(
        "--freeze_encoder",
        type=bool,
        default=True,
        help="Whether to freeze encoder layers",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=config.IMG_DIR,
        help="Directory containing images",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="smoke-segmentation.v5i.coco-segmentation/image_level_labels.json",
        help="Path to image-level labels JSON",
    )
    parser.add_argument(
        "--class_map",
        type=str,
        default="smoke-segmentation.v5i.coco-segmentation/class_map.json",
        help="Path to class mapping JSON",
    )
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)

    # 生成模型名称和保存路径
    model_name = f"UNetPlusPlus_Classifier_{config.TIMESTAMP}"
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")

    # 创建目录
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Model will be saved as: {model_path}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        args.img_dir,
        args.label_path,
        args.class_map,
        IMG_HEIGHT,
        IMG_WIDTH,
        BATCH_SIZE,
        VALIDATION_SPLIT,
    )

    # 构建模型
    model, criterion, optimizer = build_model_and_optimizer(
        pretrained_encoder=args.pretrained,
        freeze_encoder=args.freeze_encoder,
    )

    # 初始化早停对象
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True, path=model_path)

    # 训练跟踪
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # 开始训练循环
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # 训练阶段
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        # 验证阶段
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # 应用早停
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            # 如果没有验证集，直接保存模型
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch+1}")

    # 可视化训练过程
    plot_training_curves(history, RESULTS_DIR, model_name)

    print("\nTraining finished.")
    if val_loader:
        print(f"Best validation loss achieved: {early_stopping.val_loss_min:.4f}")
        print(f"Best model saved to {model_path}")
    else:
        print(f"Final model saved to {model_path}")


if __name__ == "__main__":
    main()
