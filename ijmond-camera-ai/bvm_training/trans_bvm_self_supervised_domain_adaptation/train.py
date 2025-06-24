#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
领域自适应训练脚本
整合半监督学习与领域自适应功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from domain_adapt import create_domain_adaptive_model, compute_domain_loss
from dataloader import get_loader, get_dataset_name_from_path, get_train_val_loaders
from utils import (
    AvgMeter,
    EarlyStopping,
    validate_model,
    generate_checkpoint_filename,
    generate_best_model_filename,
    generate_domain_adaptation_model_name,
    l2_regularisation,
)
from lscloss import *
from itertools import cycle
from cont_loss import intra_inter_contrastive_loss

# Define computation device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def argparser():
    parser = argparse.ArgumentParser(description="Domain Adaptive Training Script")

    # ================================== 基础训练配置 ==================================
    parser.add_argument("--epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=6, help="batch size for training")
    parser.add_argument("--trainsize", type=int, default=352, help="input image resolution (trainsize x trainsize)")

    # ================================== 优化器配置 ==================================
    parser.add_argument("--lr_gen", type=float, default=2.5e-4, help="learning rate for generator")
    parser.add_argument("--beta", type=float, default=0.5, help="beta parameter for Adam optimizer")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping threshold")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="learning rate decay factor for ReduceLROnPlateau")
    parser.add_argument("--decay_epoch", type=int, default=6, help="patience epochs for ReduceLROnPlateau scheduler")

    # ================================== 模型架构配置 ==================================
    parser.add_argument("--gen_reduced_channel", type=int, default=32, help="reduced channel count in generator")
    parser.add_argument("--feat_channel", type=int, default=32, help="feature channel count for saliency features")
    parser.add_argument("--latent_dim", type=int, default=3, help="latent space dimension")
    parser.add_argument("--num_filters", type=int, default=16, help="number of filters for contrastive loss layer")

    # ================================== 损失函数权重配置 ==================================
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for L2 regularization")
    parser.add_argument("--lat_weight", type=float, default=10.0, help="weight for latent loss")
    parser.add_argument("--vae_loss_weight", type=float, default=0.4, help="weight for VAE loss component")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1, help="weight for contrastive loss")

    # ================================== 半监督学习配置 ==================================
    parser.add_argument("--inter", action="store_true", default=False, help="use inter-image pixel matching (vs intra-image)")
    parser.add_argument("--no_samples", type=int, default=50, help="number of pixels for contrastive loss sampling")

    # ================================== 领域自适应配置 ==================================
    parser.add_argument("--domain_loss_weight", type=float, default=0.1, help="weight for domain adaptation loss")
    parser.add_argument("--lambda_grl_max", type=float, default=1.0, help="maximum lambda for gradient reversal layer")
    parser.add_argument("--num_domains", type=int, default=2, help="number of domains (source=0, target=1)")

    # ================================== 伪标签学习配置 ==================================
    parser.add_argument("--pseudo_loss_weight", type=float, default=0.5, help="weight for pseudo label supervision loss")

    # ================================== 数据集路径配置 ==================================
    parser.add_argument(
        "--source_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/train", help="source domain dataset path (with ground truth labels)"
    )
    parser.add_argument("--target_dataset_path", type=str, default="data/ijmond_data/test", help="target domain dataset path (with pseudo labels)")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="path to pretrained model weights")
    parser.add_argument("--save_model_path", type=str, default="models/domain_adapt", help="directory to save trained models")

    # ================================== 验证和早停配置 ==================================
    parser.add_argument("--val_split", type=float, default=0.2, help="fraction of source data used for validation (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="minimum improvement threshold for early stopping")
    parser.add_argument("--enable_validation", action="store_true", default=True, help="enable validation on source data subset")

    # ================================== 数据增强和可重现性配置 ==================================
    parser.add_argument("--aug", action="store_true", default=False, help="enable data augmentation for target domain data")
    parser.add_argument("--freeze", action="store_true", default=False, help="freeze randomness for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducible results")

    return parser.parse_args()


def structure_loss(pred, mask):
    """
    结构损失，用于评估预测的显著性图与真实显著性图之间的差异
    通过计算加权的二进制交叉熵损失和加权的IoU损失来实现
    Args:
        pred: predicted saliency map
        mask: ground truth saliency map
    Returns:
        loss: structure loss
    """
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )  # 计算加权因子，在mask与其局部均值之间的差异越大，权重越大，从而更关注边缘或过渡区域
    weighted_bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    weighted_bce_loss = (weight * weighted_bce_loss).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))  # 交集
    union = (((pred + mask - pred * mask)) * weight).sum(dim=(2, 3))  # 并集
    weighted_IoU = (inter + 1e-6) / (union + 1e-6)  # 加1e-6防止除0错误
    weighted_IoU_loss = 1 - weighted_IoU  # IoU损失，IoU越高，损失越低
    return (weighted_bce_loss + weighted_IoU_loss).mean()


## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def load_data(dataset_path, opt, aug=False, freeze=False):
    """
    加载数据集
    Args:
        dataset_path: 数据集路径
        opt: 训练选项
        aug: 是否启用数据增强
        freeze: 是否冻结随机性
    Returns:
        tuple: (train_loader, total_step)
    """
    image_root = os.path.join(dataset_path, "img/")
    gt_root = os.path.join(dataset_path, "gt/")
    trans_map_root = os.path.join(dataset_path, "trans/")

    train_loader = get_loader(
        image_root, gt_root, trans_map_root, batchsize=opt.batchsize, trainsize=opt.trainsize, aug=aug, freeze=freeze, random_seed=opt.random_seed
    )
    total_step = len(train_loader)
    return train_loader, total_step


def load_labeled_data_with_validation(dataset_path, opt, freeze=False):
    """
    加载标注数据集并进行训练/验证分割（当启用校验时使用）
    Args:
        dataset_path: 标注数据集路径
        opt: 训练选项
        freeze: 是否冻结随机性
    Returns:
        tuple: (train_loader, val_loader, train_step, val_step)
    """
    image_root = os.path.join(dataset_path, "img/")
    gt_root = os.path.join(dataset_path, "gt/")
    trans_map_root = os.path.join(dataset_path, "trans/")

    # 使用训练/验证分割的数据加载器
    train_loader, val_loader = get_train_val_loaders(
        image_root,
        gt_root,
        trans_map_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        val_split=opt.val_split,
        aug=False,  # 标注数据不使用数据增强以保持稳定性
        freeze=freeze,
        random_seed=opt.random_seed,
    )

    train_step = len(train_loader)
    val_step = len(val_loader)
    return train_loader, val_loader, train_step, val_step


def print_training_configuration(opt, device, model_name):
    """
    打印训练配置信息
    """
    print("=" * 80)
    print("DOMAIN ADAPTIVE TRAINING CONFIGURATION")
    print("=" * 80)

    # ================================== 基础配置 ==================================
    print("📋 BASIC TRAINING SETTINGS")
    print("-" * 40)
    print(f"  Training Epochs: {opt.epoch}")
    print(f"  Batch Size: {opt.batchsize}")
    print(f"  Training Image Size: {opt.trainsize}x{opt.trainsize}")
    print(f"  Device: {device}")
    print(f"  Model Name: {model_name}")

    # ================================== 优化器配置 ==================================
    print("\n⚙️  OPTIMIZER SETTINGS")
    print("-" * 40)
    print(f"  Learning Rate: {opt.lr_gen}")
    print(f"  Adam Beta: {opt.beta}")
    print(f"  Gradient Clipping: {opt.clip}")
    print(f"  LR Decay Factor: {opt.decay_rate}")
    print(f"  LR Patience (epochs): {opt.decay_epoch}")

    # ================================== 模型架构配置 ==================================
    print("\n🏗️  MODEL ARCHITECTURE")
    print("-" * 40)
    print(f"  Feature Channels: {opt.feat_channel}")
    print(f"  Latent Dimension: {opt.latent_dim}")
    print(f"  Generator Reduced Channels: {opt.gen_reduced_channel}")
    print(f"  Contrastive Layer Filters: {opt.num_filters}")

    # ================================== 损失函数权重 ==================================
    print("\n📊 LOSS FUNCTION WEIGHTS")
    print("-" * 40)
    print(f"  L2 Regularization: {opt.reg_weight}")
    print(f"  Latent Loss: {opt.lat_weight}")
    print(f"  VAE Loss: {opt.vae_loss_weight}")
    print(f"  Contrastive Loss: {opt.contrastive_loss_weight}")
    print(f"  Domain Adaptation Loss: {opt.domain_loss_weight}")
    print(f"  Pseudo Label Loss: {opt.pseudo_loss_weight}")

    # ================================== 域适应配置 ==================================
    print("\n🔄 DOMAIN ADAPTATION SETTINGS")
    print("-" * 40)
    print(f"  Number of Domains: {opt.num_domains}")
    print(f"  Gradient Reversal Lambda Max: {opt.lambda_grl_max}")
    print(f"  Pseudo Label Weight: {opt.pseudo_loss_weight}")

    # ================================== 半监督学习配置 ==================================
    print("\n🎯 SEMI-SUPERVISED LEARNING")
    print("-" * 40)
    print(f"  Contrastive Pixel Matching: {'Inter-image' if opt.inter else 'Intra-image'}")
    print(f"  Contrastive Sample Count: {opt.no_samples}")

    # ================================== 数据集配置 ==================================
    print("\n📁 DATASET CONFIGURATION")
    print("-" * 40)
    print(f"  Source Domain Path: {opt.source_dataset_path}")
    print(f"  Target Domain Path: {opt.target_dataset_path}")
    print(f"  Pretrained Weights: {opt.pretrained_weights or 'None'}")
    print(f"  Model Save Path: {opt.save_model_path}")

    # ================================== 验证和早停配置 ==================================
    print("\n✅ VALIDATION & EARLY STOPPING")
    print("-" * 40)
    print(f"  Enable Validation: {opt.enable_validation}")
    print(f"  Validation Split: {opt.val_split}")
    print(f"  Early Stopping Patience: {opt.patience}")
    print(f"  Min Delta for Improvement: {opt.min_delta}")

    # ================================== 数据增强配置 ==================================
    print("\n🔀 DATA AUGMENTATION & REPRODUCIBILITY")
    print("-" * 40)
    print(f"  Data Augmentation: {opt.aug}")
    print(f"  Freeze Randomness: {opt.freeze}")
    print(f"  Random Seed: {opt.random_seed}")
    if opt.freeze and opt.aug:
        print("  ⚠️  NOTE: Data augmentation disabled due to freeze mode")

    print("=" * 80)


opt = argparser()

# 设置随机种子
torch.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.random_seed)

# 获取数据集名称并生成模型名称
source_dataset_name = get_dataset_name_from_path(opt.source_dataset_path)
target_dataset_name = get_dataset_name_from_path(opt.target_dataset_path)

# 使用域适应专用的模型命名函数
model_name = generate_domain_adaptation_model_name(source_dataset_name, target_dataset_name, opt.pretrained_weights)

original_save_path = opt.save_model_path
opt.save_model_path = os.path.join(original_save_path, model_name)

# 打印训练配置
print_training_configuration(opt, device, model_name)

# 数据加载器
print("\n🔄 LOADING DATASETS...")

# 加载源域数据 (with or without validation split)
if opt.enable_validation:
    # 启用校验模式：使用训练/验证分割
    source_train_loader, val_loader, source_train_step, val_step = load_labeled_data_with_validation(opt.source_dataset_path, opt, freeze=opt.freeze)
    print(f"源域训练集: {source_train_step} batches")
    print(f"源域验证集: {val_step} batches")

    # 初始化早停策略 - 基于验证指标
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_val_iou = 0.0
    best_epoch = 0
    validation_enabled = True
else:
    # 非校验模式：使用所有源域数据进行训练
    source_train_loader, source_train_step = load_data(opt.source_dataset_path, opt, aug=False, freeze=opt.freeze)
    val_loader = None
    print(f"源域训练集: {source_train_step} batches")

    # 初始化早停策略 - 基于训练损失
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_val_loss = float("inf")
    best_epoch = 0
    validation_enabled = False

# 加载目标域数据
target_train_loader, target_train_step = load_data(opt.target_dataset_path, opt, aug=opt.aug, freeze=opt.freeze)
target_train_iter = cycle(target_train_loader)  # continuously iterate over the target dataset
print(f"目标域训练集: {target_train_step} batches")

# Use source data loader for main training loop
train_loader = source_train_loader
total_step = source_train_step

# 模型构建
print("构建模型...")
base_generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=opt.num_filters)

# 创建领域自适应模型
model = create_domain_adaptive_model(
    base_generator=base_generator, feat_channels=opt.feat_channel, num_domains=opt.num_domains, domain_loss_weight=opt.domain_loss_weight
)

model.to(device)

# 加载预训练权重
if opt.pretrained_weights and os.path.exists(opt.pretrained_weights):
    print(f"加载预训练权重: {opt.pretrained_weights}")
    checkpoint = torch.load(opt.pretrained_weights, map_location=device)
    # 只加载base_generator的权重
    if "generator_state_dict" in checkpoint:
        model.base_generator.load_state_dict(checkpoint["generator_state_dict"])
    else:
        model.base_generator.load_state_dict(checkpoint)
    print("预训练权重加载完成")

# 优化器和学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_gen, betas=(opt.beta, 0.999))
# 使用ReduceLROnPlateau调度器，根据损失自适应调整学习率
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # 监控损失，当损失不再下降时减少学习率
    factor=opt.decay_rate,  # 学习率衰减因子
    patience=opt.decay_epoch,  # 等待多少个epoch后如果没有改善就减少学习率
    min_lr=1e-7,  # 最小学习率
)

print(f"Learning Rate Scheduler configured:")
print(f"  - Type: ReduceLROnPlateau (adaptive based on loss)")
print(f"  - Patience (epochs to wait): {opt.decay_epoch}")
print(f"  - Decay Factor: {opt.decay_rate}")
print(f"  - Minimum LR: 1e-7")

# 损失函数
size_rates = [1]  # multi-scale training
loss_lsc = LocalSaliencyCoherence().to(device)  # 局部显著性一致性损失函数
loss_lsc_kernels_desc_defaults = [{"weight": 0.1, "xy": 3, "trans": 0.1}]
loss_lsc_radius = 2
weight_lsc = 0.01

print("Let's go!")
# 在训练开始前确保保存目录存在
save_path = opt.save_model_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created save directory: {save_path}")

for epoch in range(1, opt.epoch + 1):
    print("--" * 10 + "Epoch: {}/{}".format(epoch, opt.epoch) + "--" * 10)
    model.train()
    loss_record = AvgMeter()
    print("Learning Rate: {}".format(optimizer.param_groups[0]["lr"]))

    # 计算梯度反转层的lambda值（逐渐增加）
    p = float(epoch - 1) / opt.epoch
    lambda_grl = opt.lambda_grl_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1)

    for i, source_pack in enumerate(train_loader, start=1):
        # Load a batch from the target loader
        target_pack = next(target_train_iter)

        for rate in size_rates:
            optimizer.zero_grad()

            ### Load Data ######################################
            # Unpack source domain data
            images_src, gts_src, trans_src = source_pack
            images_src = images_src.to(device)
            gts_src = gts_src.to(device)
            trans_src = trans_src.to(device)

            # Unpack target domain data
            images_tgt, gts_tgt, trans_tgt = target_pack
            images_tgt = images_tgt.to(device)
            gts_tgt = gts_tgt.to(device)  # 目标域伪标签，参与训练
            trans_tgt = trans_tgt.to(device)

            ### Multi-scale training samples ############################
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images_src = F.interpolate(images_src, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts_src = F.interpolate(gts_src, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                trans_src = F.interpolate(trans_src, size=(trainsize, trainsize), mode="bilinear", align_corners=True)

                images_tgt = F.interpolate(images_tgt, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts_tgt = F.interpolate(gts_tgt, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                trans_tgt = F.interpolate(trans_tgt, size=(trainsize, trainsize), mode="bilinear", align_corners=True)

            ### 源域前向传播 ############################
            src_outputs = model(images_src, gts_src, training=True, lambda_grl=lambda_grl, source_domain=True)
            (
                sal_init_post_src,
                sal_ref_post_src,
                sal_init_prior_src,
                sal_ref_prior_src,
                latent_loss_src,
                output_post_src,
                output_prior_src,
                d_smoke_src,
                d_bg_src,
            ) = src_outputs

            ### 目标域前向传播 ############################
            # 目标域数据用于域适应和伪标签监督学习
            tgt_outputs = model(images_tgt, gts_tgt, training=True, lambda_grl=lambda_grl, source_domain=False)
            (
                sal_init_post_tgt,
                sal_ref_post_tgt,
                sal_init_prior_tgt,
                sal_ref_prior_tgt,
                latent_loss_tgt,
                output_post_tgt,
                output_prior_tgt,
                d_smoke_tgt,
                d_bg_tgt,
            ) = tgt_outputs

            ### 损失计算 ############################

            # 1. 源域监督损失（结构损失）
            src_sal_loss = 0.5 * (structure_loss(sal_init_post_src, gts_src) + structure_loss(sal_ref_post_src, gts_src))

            # 2. 目标域伪标签监督损失
            tgt_sal_loss = 0.5 * (structure_loss(sal_init_post_tgt, gts_tgt) + structure_loss(sal_ref_post_tgt, gts_tgt))

            # 总的分割损失
            sal_loss = src_sal_loss + opt.pseudo_loss_weight * tgt_sal_loss

            # 3. 对比损失（源域和目标域）
            cont_loss_src = intra_inter_contrastive_loss(output_post_src, gts_src, num_samples=opt.no_samples, margin=1.0, inter=opt.inter)
            cont_loss_tgt = intra_inter_contrastive_loss(output_post_tgt, gts_tgt, num_samples=opt.no_samples, margin=1.0, inter=opt.inter)
            cont_loss = cont_loss_src + opt.pseudo_loss_weight * cont_loss_tgt

            # 4. LSC损失计算（源域 + 目标域）
            # 源域LSC损失
            trans_scale_src = F.interpolate(trans_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_init_scale_src = F.interpolate(sal_init_prior_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_ref_scale_src = F.interpolate(sal_ref_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_init_scale_src = F.interpolate(sal_init_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_ref_scale_src = F.interpolate(sal_ref_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            sample_src = {"trans": trans_scale_src}

            # 源域LSC损失计算
            loss_lsc_1_src = loss_lsc(
                torch.sigmoid(pred_post_init_scale_src),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_src,
                trans_scale_src.shape[2],
                trans_scale_src.shape[3],
            )["loss"]
            loss_lsc_2_src = loss_lsc(
                torch.sigmoid(pred_post_ref_scale_src),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_src,
                trans_scale_src.shape[2],
                trans_scale_src.shape[3],
            )["loss"]
            loss_lsc_post_src = weight_lsc * (loss_lsc_1_src + loss_lsc_2_src)

            loss_lsc_3_src = loss_lsc(
                torch.sigmoid(pred_prior_init_scale_src),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_src,
                trans_scale_src.shape[2],
                trans_scale_src.shape[3],
            )["loss"]
            loss_lsc_4_src = loss_lsc(
                torch.sigmoid(pred_prior_ref_scale_src),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_src,
                trans_scale_src.shape[2],
                trans_scale_src.shape[3],
            )["loss"]
            loss_lsc_prior_src = weight_lsc * (loss_lsc_3_src + loss_lsc_4_src)

            # 目标域LSC损失
            trans_scale_tgt = F.interpolate(trans_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_init_scale_tgt = F.interpolate(sal_init_prior_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_ref_scale_tgt = F.interpolate(sal_ref_post_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_init_scale_tgt = F.interpolate(sal_init_post_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_ref_scale_tgt = F.interpolate(sal_ref_post_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            sample_tgt = {"trans": trans_scale_tgt}

            # 目标域LSC损失计算
            loss_lsc_1_tgt = loss_lsc(
                torch.sigmoid(pred_post_init_scale_tgt),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_tgt,
                trans_scale_tgt.shape[2],
                trans_scale_tgt.shape[3],
            )["loss"]
            loss_lsc_2_tgt = loss_lsc(
                torch.sigmoid(pred_post_ref_scale_tgt),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_tgt,
                trans_scale_tgt.shape[2],
                trans_scale_tgt.shape[3],
            )["loss"]
            loss_lsc_post_tgt = weight_lsc * (loss_lsc_1_tgt + loss_lsc_2_tgt)

            loss_lsc_3_tgt = loss_lsc(
                torch.sigmoid(pred_prior_init_scale_tgt),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_tgt,
                trans_scale_tgt.shape[2],
                trans_scale_tgt.shape[3],
            )["loss"]
            loss_lsc_4_tgt = loss_lsc(
                torch.sigmoid(pred_prior_ref_scale_tgt),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample_tgt,
                trans_scale_tgt.shape[2],
                trans_scale_tgt.shape[3],
            )["loss"]
            loss_lsc_prior_tgt = weight_lsc * (loss_lsc_3_tgt + loss_lsc_4_tgt)

            # 总LSC损失
            loss_lsc_post = loss_lsc_post_src + opt.pseudo_loss_weight * loss_lsc_post_tgt
            loss_lsc_prior = loss_lsc_prior_src + opt.pseudo_loss_weight * loss_lsc_prior_tgt

            # 4. L2正则化损失
            reg_loss = (
                l2_regularisation(model.base_generator.xy_encoder)
                + l2_regularisation(model.base_generator.x_encoder)
                + l2_regularisation(model.base_generator.sal_encoder)
            )
            reg_loss = opt.reg_weight * reg_loss

            # 5. 潜在损失（线性退火）
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            latent_loss = opt.lat_weight * anneal_reg * (latent_loss_src + latent_loss_tgt)

            # 6. 域判别损失
            domain_loss, domain_loss_dict = compute_domain_loss(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, images_src.size(0))
            domain_loss = opt.domain_loss_weight * domain_loss

            # VAE损失部分（包含源域和目标域）
            gen_loss_cvae = sal_loss + latent_loss + loss_lsc_post
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            # 结构损失部分（包含源域和目标域）
            gen_loss_gsnn = 0.5 * (
                structure_loss(sal_init_prior_src, gts_src)
                + structure_loss(sal_ref_post_src, gts_src)
                + opt.pseudo_loss_weight * (structure_loss(sal_init_prior_tgt, gts_tgt) + structure_loss(sal_ref_post_tgt, gts_tgt))
            )
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior

            ### 总损失 ###############################################
            total_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss + domain_loss + opt.contrastive_loss_weight * cont_loss  # type: torch.Tensor
            total_loss.backward()

            # Gradient clipping
            if opt.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

            optimizer.step()

            if rate == 1:
                loss_record.update(total_loss.data, opt.batchsize)

        # 打印训练信息 - 基于百分比打印（25%, 50%, 75%, 100%）
        progress_points = [int(total_step * 0.25), int(total_step * 0.5), int(total_step * 0.75), total_step]
        if i in progress_points:
            progress_pct = (i / total_step) * 100
            log_info = (
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}] ({:.0f}%), Total Loss: {:.4f}, "
                "Src Seg Loss: {:.4f}, Tgt Seg Loss: {:.4f}, Contrastive Loss: {:.4f}, Domain Loss: {:.4f}".format(
                    datetime.now(),
                    epoch,
                    opt.epoch,
                    i,
                    total_step,
                    progress_pct,
                    loss_record.show(),
                    src_sal_loss.item(),
                    tgt_sal_loss.item(),
                    cont_loss.item(),
                    domain_loss.item(),
                )
            )

            print(log_info)  # 在训练循环结束后调用scheduler.step() - ReduceLROnPlateau需要传入监控的指标
    old_lr = optimizer.param_groups[0]["lr"]

    # 根据是否启用验证来选择监控的指标
    if validation_enabled and val_loader is not None:
        # 如果启用验证，稍后在验证后调用scheduler.step(val_loss)
        pass
    else:
        # 如果未启用验证，使用训练损失
        current_loss = loss_record.avg
        scheduler.step(current_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    if old_lr != current_lr:
        print(f"Epoch {epoch} completed. Learning rate changed: {old_lr:.6f} -> {current_lr:.6f}")
    else:
        print(f"Epoch {epoch} completed. Learning rate: {current_lr:.6f}")

    # 校验和早停逻辑
    if validation_enabled and val_loader is not None:
        # 启用校验模式：在验证集上评估模型
        print("Starting validation...")
        val_loss, val_metrics = validate_model(model.base_generator, val_loader, device, structure_loss)

        print(f"Validation Results - Loss: {val_loss:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}")
        print(f"  F1-Score: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

        # 使用验证损失更新学习率调度器
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Learning rate adjusted after validation: {current_lr:.6f} -> {new_lr:.6f}")

        # 检查是否是最佳模型 - 使用IoU作为主要指标
        current_iou = val_metrics["iou"]
        if current_iou > best_val_iou:
            best_val_iou = current_iou
            best_epoch = epoch
            # 保存最佳模型
            best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
            torch.save(model.base_generator.state_dict(), os.path.join(opt.save_model_path, best_model_filename))
            print(f"New best model saved! Validation IoU: {current_iou:.4f}")

        # 早停检查 - 使用IoU
        early_stopping(current_iou, model.base_generator)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation IoU: {best_val_iou:.4f} achieved at epoch {best_epoch}")
            break
    else:
        # 非校验模式：使用训练损失进行早停判断
        current_loss = loss_record.avg

        # 检查是否是最佳模型
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            best_epoch = epoch
            # 保存最佳模型
            best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
            torch.save(model.base_generator.state_dict(), os.path.join(opt.save_model_path, best_model_filename))
            print(f"New best model saved! Training loss: {current_loss:.4f}")

        # 早停检查 - 使用训练损失
        early_stopping(current_loss, model.base_generator)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best training loss: {best_val_loss:.4f} achieved at epoch {best_epoch}")
            break

    # 定期保存检查点 - 使用动态文件名
    if epoch >= 0 and epoch % 10 == 0:
        checkpoint_filename = generate_checkpoint_filename(epoch, model_name, opt.pretrained_weights)
        torch.save(model.base_generator.state_dict(), os.path.join(opt.save_model_path, checkpoint_filename))
        print(f"Checkpoint saved: {checkpoint_filename}")

# 训练结束后的总结
print("\n" + "=" * 50)
print("Two-Domain Adaptive Training with Pseudo Labels completed!")
print(f"Source Domain: {source_dataset_name} (Ground Truth Labels)")
print(f"Target Domain: {target_dataset_name} (Pseudo Labels)")
print(f"Pseudo Label Weight: {opt.pseudo_loss_weight}")
if validation_enabled:
    print(f"Best validation IoU: {best_val_iou:.4f} achieved at epoch {best_epoch}")
else:
    print(f"Best training loss: {best_val_loss:.4f} achieved at epoch {best_epoch}")
best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
print(f"Best model saved at: {os.path.join(opt.save_model_path, best_model_filename)}")
print("=" * 50)
