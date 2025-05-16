#!/usr/bin/env python3
# 重构为模块化的训练脚本，支持多种模型类型
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import os
import numpy as np
import argparse
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import pandas as pd
from torch.amp import autocast, GradScaler

# 从本地文件导入
from dataset import SmokeDataset_Seg
from UNetPlusPlus import UNetPlusPlus_Segmentor
from UNetPlusPlus_CBAM import UNetPlusPlus_CBAM_Segmentor
from losses import dice_loss
from metrics import calculate_metrics
from utils import plot_training_curves, EarlyStopping
import config

# --- 配置参数 ---
MODEL_REGISTRY = {
    "UNetPlusPlus": UNetPlusPlus_Segmentor,
    "UNetPlusPlus_CBAM": UNetPlusPlus_CBAM_Segmentor,
}


# --- 模块化函数定义 ---
def get_dataloaders(img_dir, mask_dir, img_height, img_width, batch_size, val_split):
    # 对图像的变换
    img_transform = T.Compose(
        [
            T.Resize((img_height, img_width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 对掩码的变换 (只需要缩放)
    mask_transform = T.Compose(
        [
            T.Resize((img_height, img_width), interpolation=T.InterpolationMode.NEAREST),
        ]
    )

    try:
        full_dataset = SmokeDataset_Seg(
            img_dir=img_dir,
            mask_dir=mask_dir,
            transform=img_transform,
            target_transform=mask_transform,
        )
    except RuntimeError as e:
        print(f"Error initializing dataset: {e}")
        exit()

    # 划分训练集和验证集
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    # Add check for small datasets
    if n_train == 0 or n_val == 0:
        print(
            f"Warning: Dataset size ({len(full_dataset)}) is too small for the validation split ({val_split}). Adjust split or dataset size."
        )
        if n_train == 0:
            print("Error: Training set size is zero.")
            exit()
        print("Using full dataset for training, no validation.")
        train_dataset = full_dataset
        val_dataset = None
    else:
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 验证集加载器，如果存在，则创建，否则设置为None
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
        raise ValueError("Validation dataset is empty. No validation will be performed.")

    return train_loader, val_loader


def build_model_and_optimizer(model_class, device, learning_rate, weight_decay, num_classes):
    model = model_class(n_channels=3, n_classes=num_classes, deep_supervision=True).to(device)

    # 使用加权交叉熵
    class_weights = torch.tensor([1.0, 15.0, 10.0], device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using Weighted Cross Entropy with manual weights.")

    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 初始化混合精度训练的GradScaler
    scaler = GradScaler(device, enabled=(device == "cuda"))
    print(f"AMP Enabled: {scaler.is_enabled()}")

    # 添加ReduceLROnPlateau调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=config.LR_SCHEDULER_PATIENCE, factor=config.LR_SCHEDULER_FACTOR
    )
    return model, criterion_ce, optimizer, scheduler, scaler


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    epoch_train_loss = 0.0
    epoch_train_acc = 0.0
    epoch_train_iou = 0.0
    all_preds_train = []
    all_masks_train = []

    for i, (inputs, masks) in enumerate(loader):
        inputs = inputs.to(device)
        masks = masks.to(device)

        # 零梯度
        optimizer.zero_grad(set_to_none=True)

        # 混合精度上下文
        with autocast(device_type=device, enabled=scaler.is_enabled()):
            outputs = model(inputs)
            if isinstance(outputs, list):
                # 训练时对每个输出都计算损失，取平均
                loss = 0
                for out in outputs:
                    loss += criterion(out, masks)
                    loss += dice_loss(out, masks, config.NUM_CLASSES)
                loss = loss / (2 * len(outputs))
                # 后处理时用最后一个输出
                final_output = outputs[-1]
            else:
                loss_ce = criterion(outputs, masks)
                loss_d = dice_loss(outputs, masks, config.NUM_CLASSES)
                loss = config.COMBINED_LOSS_WEIGHT_CE * loss_ce + config.COMBINED_LOSS_WEIGHT_DICE * loss_d
                final_output = outputs

        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 累加损失和指标
        epoch_train_loss += loss.item()
        acc, iou = calculate_metrics(final_output.detach(), masks.detach(), config.NUM_CLASSES)
        epoch_train_acc += acc
        epoch_train_iou += iou

        # 存储预测和掩码用于计算混淆矩阵
        preds_train = torch.argmax(final_output, dim=1)
        all_preds_train.append(preds_train.cpu().numpy().flatten())
        all_masks_train.append(masks.cpu().numpy().flatten())

    # 计算平均指标
    avg_train_loss = epoch_train_loss / len(loader)
    avg_train_acc = epoch_train_acc / len(loader)
    avg_train_iou = epoch_train_iou / len(loader)

    # 训练集recall/f1
    all_preds_np = np.concatenate(all_preds_train)
    all_masks_np = np.concatenate(all_masks_train)
    train_recall = recall_score(all_masks_np, all_preds_np, average="macro", zero_division=0)
    train_f1 = f1_score(all_masks_np, all_preds_np, average="macro", zero_division=0)

    print(
        f"Training   - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f}, Mean IoU: {avg_train_iou:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}"
    )

    return avg_train_loss, avg_train_acc, avg_train_iou, train_recall, train_f1


def validate_one_epoch(
    model,
    loader,
    criterion,
    scaler,
    device,
    class_names,
    checkpoint_dir,
    model_name_base,
    epoch,
    best_model_path,
):
    """验证一个epoch并返回验证指标"""
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_acc = 0.0
    epoch_val_iou = 0.0
    all_preds_val = []
    all_masks_val = []
    with torch.no_grad():
        for inputs, masks in loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            with autocast(device_type=device, enabled=scaler.is_enabled()):
                outputs = model(inputs)
                if isinstance(outputs, list):
                    loss = 0
                    for out in outputs:
                        loss += criterion(out, masks)
                        loss += dice_loss(out, masks, config.NUM_CLASSES)
                    loss = loss / (2 * len(outputs))
                    final_output = outputs[-1]
                else:
                    loss_ce = criterion(outputs, masks)
                    loss_d = dice_loss(outputs, masks, config.NUM_CLASSES)
                    loss = config.COMBINED_LOSS_WEIGHT_CE * loss_ce + config.COMBINED_LOSS_WEIGHT_DICE * loss_d
                    final_output = outputs

            epoch_val_loss += loss.item()
            acc, iou = calculate_metrics(final_output, masks, config.NUM_CLASSES)
            epoch_val_acc += acc
            epoch_val_iou += iou

            preds_val = torch.argmax(final_output, dim=1)
            all_preds_val.append(preds_val.cpu().numpy().flatten())
            all_masks_val.append(masks.cpu().numpy().flatten())

    avg_val_loss = epoch_val_loss / len(loader)
    avg_val_acc = epoch_val_acc / len(loader)
    avg_val_iou = epoch_val_iou / len(loader)

    # 计算并打印混淆矩阵
    all_preds_np = np.concatenate(all_preds_val)
    all_masks_np = np.concatenate(all_masks_val)
    cm = confusion_matrix(all_preds_np, all_masks_np, labels=list(range(config.NUM_CLASSES)))

    print("Validation Confusion Matrix:")
    try:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print(cm_df)
    except ImportError:
        print("(Install pandas for better formatting)")
        print(cm)

    # 提取烟雾像素数量
    tp_high_smoke = cm[1, 1]
    fn_high_smoke = cm[1, 0] + cm[1, 2]
    tp_low_smoke = cm[2, 2]
    fn_low_smoke = cm[2, 0] + cm[2, 1]
    total_actual_high_smoke = tp_high_smoke + fn_high_smoke
    total_actual_low_smoke = tp_low_smoke + fn_low_smoke

    val_recall = recall_score(all_masks_np, all_preds_np, average="macro", zero_division=0)
    val_f1 = f1_score(all_masks_np, all_preds_np, average="macro", zero_division=0)

    print(
        f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}, Mean IoU: {avg_val_iou:.4f}, "
        f"Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
    )
    print(
        f"  High Opacity Smoke Pixels: Correctly Predicted (TP): {tp_high_smoke} / Actual Total: {total_actual_high_smoke}"
    )
    print(
        f"  Low Opcaity Smoke Pixels:  Correctly Predicted (TP): {tp_low_smoke} / Actual Total: {total_actual_low_smoke}"
    )

    # 保存检查点
    checkpoint_filename = f"{model_name_base}_epoch_{epoch+1}_val_loss_{avg_val_loss:.4f}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    torch.save(model.state_dict(), checkpoint_path)

    return avg_val_loss, avg_val_acc, val_recall, val_f1


def main():
    parser = argparse.ArgumentParser(description="Train Smoke Segmentation Model")
    parser.add_argument(
        "--model",
        type=str,
        default="unetplusplus",
        choices=MODEL_REGISTRY.keys(),
        help="Model architecture to use (default: unetplusplus)",
    )
    args = parser.parse_args()

    # 根据命令行参数选择模型类
    model_class = MODEL_REGISTRY[args.model]

    # 生成模型名称和保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_base = f"{args.model}_smoke_{timestamp}"
    best_model_path = os.path.join(config.MODELS_DIR, f"{model_name_base}_best.pth")
    final_model_path = os.path.join(config.MODELS_DIR, f"{model_name_base}_final.pth")

    # 创建保存目录
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Using model: {args.model}")
    print(f"Using device: {config.DEVICE}")
    print(f"Model will be saved as: {model_name_base}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")

    train_loader, val_loader = get_dataloaders(
        config.IMG_DIR,
        config.MASK_DIR,
        config.IMG_HEIGHT,
        config.IMG_WIDTH,
        config.BATCH_SIZE,
        config.VALIDATION_SPLIT,
    )
    model, criterion_ce, optimizer, scheduler, scaler = build_model_and_optimizer(
        model_class, config.DEVICE, config.LEARNING_RATE, config.WEIGHT_DECAY, config.NUM_CLASSES
    )

    # 初始化早停对象
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True, path=best_model_path)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        # 训练阶段
        avg_train_loss, avg_train_acc, avg_train_iou, train_recall, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion_ce, config.DEVICE
        )
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["train_recall"].append(train_recall)
        history["train_f1"].append(train_f1)

        # 验证阶段
        if val_loader:
            avg_val_loss, avg_val_acc, val_recall, val_f1 = validate_one_epoch(
                model,
                val_loader,
                criterion_ce,
                scaler,
                config.DEVICE,
                config.CLASS_NAMES,
                config.CHECKPOINT_DIR_PATH,
                model_name_base,
                epoch,
                best_model_path,
            )

            # 更新学习率
            if scheduler is not None:
                # 记录之前的学习率
                prev_lr = optimizer.param_groups[0]["lr"]
                # 调用步进
                scheduler.step(avg_val_loss)
                # 获取当前学习率
                curr_lr = optimizer.param_groups[0]["lr"]
                # 如果学习率变化，打印信息
                if curr_lr != prev_lr:
                    print(f"Learning rate changed from {prev_lr:.6f} to {curr_lr:.6f}")

            # 更新历史记录
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(avg_val_acc)
            history["val_recall"].append(val_recall)
            history["val_f1"].append(val_f1)

            # 应用早停
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # 训练结束后可视化训练过程
    plot_training_curves(
        history,
        save_dir=config.RESULTS_DIR,
        model_name=model_name_base,
        timestamp=timestamp,
    )

    # 最终保存逻辑
    if not early_stopping.early_stop and epoch == config.NUM_EPOCHS - 1:
        # 完成所有轮次但未触发早停，保存最终模型
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path} (completed all epochs)")

    print("\nTraining finished.")
    if val_loader:
        print(f"Best validation loss achieved: {early_stopping.val_loss_min:.4f}")
        print(f"Best model state dict saved to {best_model_path}")
    elif os.path.exists(final_model_path):
        print(f"Final model state dict saved to {final_model_path}")
    else:
        print("Training finished without saving a final model. Checkpoints might be available.")


if __name__ == "__main__":
    main()
