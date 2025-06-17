import torch
import torch.nn.functional as F
import warnings
from torch import no_grad

import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator, Descriptor
from dataloader import get_train_val_loaders, get_dataset_name_from_path
from utils import adjust_lr, AvgMeter, EarlyStopping, validate_model, generate_model_name, generate_checkpoint_filename, generate_best_model_filename
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
import smoothness
from lscloss import *

# Define computation device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_argparser():
    parser = argparse.ArgumentParser(description="Fully Supervised Training Script")

    # ================================== 基础训练配置 ==================================
    parser.add_argument("--epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=7, help="batch size for training")
    parser.add_argument("--trainsize", type=int, default=352, help="input image resolution (trainsize x trainsize)")

    # ================================== 优化器配置 ==================================
    parser.add_argument("--lr_gen", type=float, default=2.5e-5, help="learning rate for generator")
    parser.add_argument("--lr_des", type=float, default=2.5e-5, help="learning rate for descriptor")
    parser.add_argument("--beta", type=float, default=0.5, help="beta parameter for Adam optimizer")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping threshold")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="learning rate decay factor for ReduceLROnPlateau")
    parser.add_argument("--decay_epoch", type=int, default=6, help="patience epochs for ReduceLROnPlateau scheduler")

    # ================================== 模型架构配置 ==================================
    parser.add_argument("--gen_reduced_channel", type=int, default=32, help="reduced channel count in generator")
    parser.add_argument("--des_reduced_channel", type=int, default=64, help="reduced channel count in descriptor")
    parser.add_argument("--feat_channel", type=int, default=32, help="feature channel count for saliency features")
    parser.add_argument("--latent_dim", type=int, default=3, help="latent space dimension")

    # ================================== EBM模型配置 ==================================
    parser.add_argument("--langevin_step_num_des", type=int, default=10, help="number of Langevin steps for EBM")
    parser.add_argument("--langevin_step_size_des", type=float, default=0.026, help="step size for EBM Langevin sampling")
    parser.add_argument(
        "--energy_form", type=str, default="identity", choices=["tanh", "sigmoid", "identity", "softplus"], help="energy function form"
    )

    # ================================== 损失函数权重配置 ==================================
    parser.add_argument("--sm_weight", type=float, default=0.1, help="weight for smoothness loss")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for L2 regularization")
    parser.add_argument("--lat_weight", type=float, default=10.0, help="weight for latent loss")
    parser.add_argument("--vae_loss_weight", type=float, default=0.4, help="weight for VAE loss component")

    # ================================== 数据集路径配置 ==================================
    parser.add_argument("--dataset_path", type=str, default="data/ijmond_data/train", help="training dataset path")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="path to pretrained model weights")
    parser.add_argument("--save_model_path", type=str, default="models/full-supervision", help="directory to save trained models")

    # ================================== 验证和早停配置 ==================================
    parser.add_argument("--val_split", type=float, default=0.2, help="fraction of dataset used for validation (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="minimum improvement threshold for early stopping")

    # ================================== 数据增强和可重现性配置 ==================================
    parser.add_argument("--aug", action="store_true", default=False, help="enable data augmentation for training")
    parser.add_argument("--freeze", action="store_true", default=False, help="freeze randomness for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducible results")

    return parser


def print_training_configuration(opt, device, dataset_name, model_name, original_save_path):
    """
    打印训练配置信息
    """
    print("=" * 80)
    print("FULLY SUPERVISED TRAINING CONFIGURATION")
    print("=" * 80)

    # ================================== 基础配置 ==================================
    print("📋 BASIC TRAINING SETTINGS")
    print("-" * 40)
    print(f"  Training Epochs: {opt.epoch}")
    print(f"  Batch Size: {opt.batchsize}")
    print(f"  Training Image Size: {opt.trainsize}x{opt.trainsize}")
    print(f"  Device: {device}")
    print(f"  Dataset Name: {dataset_name}")
    print(f"  Model Name: {model_name}")

    # ================================== 优化器配置 ==================================
    print("\n⚙️  OPTIMIZER SETTINGS")
    print("-" * 40)
    print(f"  Generator Learning Rate: {opt.lr_gen}")
    print(f"  Descriptor Learning Rate: {opt.lr_des}")
    print(f"  Adam Beta: {opt.beta}")
    print(f"  Gradient Clipping: {opt.clip}")
    print(f"  LR Decay Factor: {opt.decay_rate}")
    print(f"  LR Patience (epochs): {opt.decay_epoch}")

    # ================================== 模型架构配置 ==================================
    print("\n🏗️  MODEL ARCHITECTURE")
    print("-" * 40)
    print(f"  Generator Reduced Channels: {opt.gen_reduced_channel}")
    print(f"  Descriptor Reduced Channels: {opt.des_reduced_channel}")
    print(f"  Feature Channels: {opt.feat_channel}")
    print(f"  Latent Dimension: {opt.latent_dim}")

    # ================================== EBM配置 ==================================
    print("\n⚡ ENERGY-BASED MODEL SETTINGS")
    print("-" * 40)
    print(f"  Langevin Steps: {opt.langevin_step_num_des}")
    print(f"  Langevin Step Size: {opt.langevin_step_size_des}")
    print(f"  Energy Function Form: {opt.energy_form}")

    # ================================== 损失函数权重 ==================================
    print("\n📊 LOSS FUNCTION WEIGHTS")
    print("-" * 40)
    print(f"  Smoothness Loss: {opt.sm_weight}")
    print(f"  L2 Regularization: {opt.reg_weight}")
    print(f"  Latent Loss: {opt.lat_weight}")
    print(f"  VAE Loss: {opt.vae_loss_weight}")

    # ================================== 数据集配置 ==================================
    print("\n📁 DATASET CONFIGURATION")
    print("-" * 40)
    print(f"  Dataset Path: {opt.dataset_path}")
    print(f"  Pretrained Weights: {opt.pretrained_weights or 'None'}")
    print(f"  Original Save Path: {original_save_path}")
    print(f"  Final Save Path: {opt.save_model_path}")

    # ================================== 验证和早停配置 ==================================
    print("\n✅ VALIDATION & EARLY STOPPING")
    print("-" * 40)
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


parser = create_argparser()
parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")  # 随机种子

# aug	freeze	效果	适用场景
# ❌	❌	基础训练，无增强	快速测试
# ✅	❌	正常训练，有增强	推荐训练
# ❌	✅	调试模式，完全固定	调试模型
# ✅	✅	调试模式，禁用增强	调试增强逻辑


# 所有超参数保存在opt中
opt = parser.parse_args()

# 获取数据集名称并生成模型名称
dataset_name = get_dataset_name_from_path(opt.dataset_path)
model_name = generate_model_name(dataset_name, opt.pretrained_weights)
original_save_path = opt.save_model_path
opt.save_model_path = os.path.join(original_save_path, model_name)

# 打印训练配置
print_training_configuration(opt, device, dataset_name, model_name, original_save_path)
print("  - Energy Form: {}".format(opt.energy_form))
print("\nData Augmentation & Reproducibility:")
print("  - Data Augmentation: {}".format("Enabled" if opt.aug else "Disabled"))
print("  - Freeze Mode: {}".format("Enabled" if opt.freeze else "Disabled"))
print("  - Random Seed: {}".format(opt.random_seed))
if opt.freeze:
    print("  - [WARNING] Freeze mode enabled - all randomness frozen for debugging")
if opt.freeze and opt.aug:
    print("  - [INFO] Data augmentation will be disabled due to freeze mode")
print("==========================================\n")

# build models
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)  # 生成器模型

# 如果有预训练权重，则加载预训练权重，否则使用随机初始化
if opt.pretrained_weights is not None:
    print(f"Load pretrained weights: {opt.pretrained_weights}")
    generator.load_state_dict(torch.load(opt.pretrained_weights))

generator.to(device)  # 将生成器模型移动到计算设备上
generator_params = generator.parameters()  # 获取生成器模型的参数，格式为可迭代对象
generator_optimizer = torch.optim.Adam(
    generator_params, lr=opt.lr_gen, betas=(opt.beta, 0.999)
)  # Adam优化器，betas的作用是控制一阶矩估计和二阶矩估计的衰减率

image_root = os.path.join(opt.dataset_path, "img/")  # data/ijmond_data/test/img
gt_root = os.path.join(opt.dataset_path, "gt/")  # data/ijmond_data/test/gt
trans_map_root = os.path.join(opt.dataset_path, "trans/")  # data/ijmond_data/test/trans

# 获取数据加载器 - 使用新的数据增强和可重现性参数
train_loader, val_loader = get_train_val_loaders(
    image_root,
    gt_root,
    trans_map_root,
    batchsize=opt.batchsize,
    trainsize=opt.trainsize,
    val_split=opt.val_split,
    aug=opt.aug,
    freeze=opt.freeze,
    random_seed=opt.random_seed,
)
# 计算数据集的总步数，训练集被分成多个batch进行训练
total_step = len(train_loader)
print(f"Training steps per epoch: {total_step}")
print(f"Validation steps per epoch: {len(val_loader)}")

# 初始化早停策略和最佳模型跟踪
early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
best_val_iou = 0.0
best_epoch = 0

# 学习率调度器 - 使用ReduceLROnPlateau调度器，根据损失自适应调整学习率
scheduler = lr_scheduler.ReduceLROnPlateau(
    generator_optimizer,
    mode="min",  # 监控损失，当损失不再下降时减少学习率
    factor=opt.decay_rate,  # 学习率衰减因子
    patience=opt.decay_epoch,  # 等待多少个epoch后如果没有改善就减少学习率
    min_lr=1e-7,  # 最小学习率
)
print(f"Learning Rate Scheduler configured:")
print(f"  - Type: ReduceLROnPlateau (adaptive based on validation loss)")
print(f"  - Patience (epochs to wait): {opt.decay_epoch}")
print(f"  - Decay Factor: {opt.decay_rate}")
print(f"  - Minimum LR: 1e-7")

bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(reduction="mean")  # 新版PyTorch使用reduction参数
size_rates = [1]  # multi-scale training，尺度因子，这里设置为1表示不进行缩放
smooth_loss = smoothness.smoothness_loss(size_average=True)  # 平滑性损失函数，约束生成的图像平滑性
lsc_loss = LocalSaliencyCoherence().to(device)  # 局部显著性一致性损失函数，在细粒度区域加强预测的一致性
lsc_loss_kernels_desc_defaults = [{"weight": 0.1, "xy": 3, "trans": 0.1}]  # 用于计算核函数
lsc_loss_radius = 2  # 邻域半径
weight_lsc = 0.01  # 控制局部显著性一致性损失在总损失中的权重


def structure_loss(pred, mask):
    """
    结构损失，用于评估预测的显著性图与真实显著性图之间的差异
    通过计算加权的二进制交叉熵损失和加权的IoU损失来实现
    Args:
        pred: predicted saliency map
        mask: ground truth saliency map
    Returns:
        loss: structure loss
    Formula:
        IoU = (pred * mask) / (pred + mask - pred * mask)
        loss = (BCE(pred, mask) + (1 - IoU)) / 2
    """
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )  # 计算加权因子，在mask与其局部均值之间的差异越大，权重越大，从而更关注边缘或过渡区域
    weighted_bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    weighted_bce_loss = (weight * weighted_bce_loss).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))  # dim=(2, 3)表示在空间维度上进行求和，对应高度和宽度

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))  # 交集，只有在对应像素处两者都较高时（即都有较高置信度）乘积才大，反映了共同激活区域的强度
    union = (((pred + mask - pred * mask)) * weight).sum(dim=(2, 3))  # 表示预测和真实各自的贡献
    weighted_IoU = (inter + 1e-6) / (union + 1e-6)  # 加1e-6防止除0错误
    weighted_IoU_loss = 1 - weighted_IoU  # IoU损失，IoU越高，损失越低
    return (weighted_bce_loss + weighted_IoU_loss).mean()


def visualize_prediction_init(pred):
    """
    可视化预测结果
    Args:
        pred: Predicted saliency map, size: [batch_size, channels, height, width]
    """
    # 遍历每个batch中的图像
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]  # 提取第kk个图像的预测结果
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0  # 将预测结果缩放到0-255范围
        pred_edge_kk = pred_edge_kk.astype(np.uint8)  # 转换为uint8类型
        save_path = "./temp/"
        name = "{:02d}_init.png".format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def visualize_prediction_ref(pred):

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_ref.png".format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_gt.png".format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def visualize_original_img(rec_img):
    img_transform = transforms.Compose(
        [transforms.Normalize(mean=[-0.4850 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
    )
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk, :, :, :]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = "./temp/"
        name = "{:02d}_img.png".format(kk)
        current_img = current_img.transpose((1, 2, 0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path + name, new_img)


## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """
    Linear annealing of a parameter.
    Args:
        init: initial value
        fin: final value
        step: current step
        annealing_steps: total steps for annealing
    """
    if annealing_steps == 0:  # 如果没有设置退火步数，则直接返回最终值
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


print("Let's go!")
# 在训练开始前确保保存目录存在
save_path = opt.save_model_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created save directory: {save_path}")

for epoch in range(1, (opt.epoch + 1)):
    print("--" * 10 + "Epoch: {}/{}".format(epoch, opt.epoch) + "--" * 10)
    # 移除此处的scheduler.step()，将在epoch结束后调用
    generator.train()
    loss_record = AvgMeter()

    # 训练阶段
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, trans = pack
            # 使用设备无关的.to(device)替代.cuda()
            images = images.to(device)
            gts = gts.to(device)
            trans = trans.to(device)
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)  # 将训练大小调整为32的倍数，兼容大多数网络（上下采样操作需要输入尺寸为32的倍数）
            if rate != 1:  # 如果不是原始大小，则进行上采样
                images = F.interpolate(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                trans = F.interpolate(trans, size=(trainsize, trainsize), mode="bilinear", align_corners=True)

            pred_post_init, pred_post_ref, pred_prior_init, pred_piror_ref, latent_loss = generator.forward(images, gts)

            # re-scale data for crf loss
            # 下采样至原来的0.3倍，方便计算CRF损失，提升计算速度
            trans_scale = F.interpolate(trans, scale_factor=0.3, mode="bilinear", align_corners=True)
            images_scale = F.interpolate(images, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_init_scale = F.interpolate(pred_prior_init, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_ref_scale = F.interpolate(pred_post_ref, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_init_scale = F.interpolate(pred_post_init, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_ref_scale = F.interpolate(pred_post_ref, scale_factor=0.3, mode="bilinear", align_corners=True)
            sample = {"trans": trans_scale}

            loss_lsc_1 = lsc_loss(
                torch.sigmoid(pred_post_init_scale),
                lsc_loss_kernels_desc_defaults,
                lsc_loss_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_2 = lsc_loss(
                torch.sigmoid(pred_post_ref_scale),
                lsc_loss_kernels_desc_defaults,
                lsc_loss_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_post = weight_lsc * (loss_lsc_1 + loss_lsc_2)

            ## l2 regularizer the inference model
            reg_loss = l2_regularisation(generator.xy_encoder) + l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
            # smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), grays)
            reg_loss = opt.reg_weight * reg_loss  # 对正则化损失进行加权，控制正则化损失在总损失中的权重
            latent_loss = latent_loss

            sal_loss = 0.5 * (
                structure_loss(pred_post_init, gts) + structure_loss(pred_post_ref, gts)
            )  # 两个预测后验结果的结构损失，衡量预测结果与真实标签之间的差异，兼顾像素级和区域级的准确性
            anneal_reg = linear_annealing(
                0, 1, epoch, opt.epoch
            )  # 防止训练初期潜在空间崩塌（posterior collapse），让模型先关注重建，再逐步加强对潜在分布的约束
            latent_loss = opt.lat_weight * anneal_reg * latent_loss

            loss_lsc_3 = lsc_loss(
                torch.sigmoid(pred_prior_init_scale),
                lsc_loss_kernels_desc_defaults,
                lsc_loss_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_4 = lsc_loss(
                torch.sigmoid(pred_prior_ref_scale),
                lsc_loss_kernels_desc_defaults,
                lsc_loss_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_prior = weight_lsc * (loss_lsc_3 + loss_lsc_4)

            # 条件自分自编码器损失，包括显著性差异损失、潜在空间损失和后验一致性损失
            gen_loss_cvae = sal_loss + latent_loss + loss_lsc_post
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            # 生成器的结构损失
            gen_loss_gsnn = 0.5 * (structure_loss(pred_prior_init, gts) + structure_loss(pred_post_ref, gts))
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior
            # total loss
            gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss
            gen_loss.backward()
            generator_optimizer.step()

            if rate == 1:
                loss_record.update(gen_loss.data, opt.batchsize)

        # 打印训练信息 - 基于百分比打印（25%, 50%, 75%, 100%）
        progress_points = [int(total_step * 0.25), int(total_step * 0.5), int(total_step * 0.75), total_step]
        if i in progress_points:
            progress_pct = (i / total_step) * 100
            # 计算像素级混淆矩阵指标
            with torch.no_grad():
                # 二值化预测，阈值0.5
                pred_bin = (torch.sigmoid(pred_post_init) > 0.5).float()
                gt_bin = gts
                # 展平所有像素
                pred_flat = pred_bin.view(-1)
                gt_flat = gt_bin.view(-1)
                tp = ((pred_flat == 1) & (gt_flat == 1)).sum().item()
                tn = ((pred_flat == 0) & (gt_flat == 0)).sum().item()
                fp = ((pred_flat == 1) & (gt_flat == 0)).sum().item()
                fn = ((pred_flat == 0) & (gt_flat == 1)).sum().item()
            # 打印总损失及混淆矩阵
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}] ({:.0f}%), Gen Loss: {:.4f}, TP: {}, FP: {}, TN: {}, FN: {}".format(
                    datetime.now(), epoch, opt.epoch, i, total_step, progress_pct, loss_record.show(), tp, fp, tn, fn
                )
            )

    # 校验阶段
    print("Starting validation...")
    val_loss, val_metrics = validate_model(generator, val_loader, device, structure_loss)

    print(f"Validation Results - Loss: {val_loss:.4f}")
    print(f"  IoU: {val_metrics['iou']:.4f}")
    print(f"  F1-Score: {val_metrics['f1']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

    # 在验证后调用scheduler.step() - ReduceLROnPlateau需要传入监控的指标
    old_lr = generator_optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)  # 使用验证损失更新学习率
    current_lr = generator_optimizer.param_groups[0]["lr"]

    if old_lr != current_lr:
        print(f"Epoch {epoch} completed. Learning rate changed: {old_lr:.6f} -> {current_lr:.6f}")
    else:
        print(f"Epoch {epoch} completed. Learning rate: {current_lr:.6f}")

    # 检查是否是最佳模型 - 使用IoU作为主要指标
    current_iou = val_metrics["iou"]
    current_f1 = val_metrics["f1"]
    if current_iou > best_val_iou:
        best_val_iou = current_iou
        best_epoch = epoch
        # 保存最佳模型 - 使用动态文件名
        save_path = opt.save_model_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
        best_model_path = os.path.join(save_path, best_model_filename)
        torch.save(generator.state_dict(), best_model_path)
        print(f"🎉 New best model saved! IoU: {current_iou:.4f}, F1: {current_f1:.4f}")
        print(f"   Saved as: {best_model_filename}")

    # 早停检查 - 使用IoU
    early_stopping(current_iou, generator)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        print(f"Best IoU score: {best_val_iou:.4f} at epoch {best_epoch}")
        break

    # 定期保存检查点 - 使用动态文件名
    save_path = opt.save_model_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch >= 0 and epoch % 10 == 0:
        checkpoint_filename = generate_checkpoint_filename(epoch, model_name, opt.pretrained_weights)
        checkpoint_path = os.path.join(save_path, checkpoint_filename)
        torch.save(generator.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_filename}")

# 训练结束后的总结
print("\n" + "=" * 50)
print("Training completed!")
print(f"Best validation IoU score: {best_val_iou:.4f} achieved at epoch {best_epoch}")
best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
print(f"Best model saved at: {os.path.join(opt.save_model_path, best_model_filename)}")
print("=" * 50)
