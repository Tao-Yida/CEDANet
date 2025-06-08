#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
领域自适应训练脚本
集成域判别器到现有的Trans_BVM训练流程中
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
import torchvision.transforms as transforms
import smoothness
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from domain_adapt import create_domain_adaptive_model, compute_domain_loss, log_domain_adaptation_stats
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
from utils import l2_regularisation
from lscloss import *
from itertools import cycle
from cont_loss import intra_inter_contrastive_loss
from PIL import Image


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--lr_gen", type=float, default=2.5e-5, help="learning rate for generator")
    parser.add_argument("--batchsize", type=int, default=6, help="training batch size")
    parser.add_argument("--trainsize", type=int, default=352, help="training dataset size")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="decay rate of learning rate")
    parser.add_argument("--decay_epoch", type=int, default=20, help="every n epochs decay learning rate")
    parser.add_argument("--beta", type=float, default=0.5, help="beta of Adam for generator")
    parser.add_argument("--gen_reduced_channel", type=int, default=32, help="reduced channel in generator")
    parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")
    parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
    parser.add_argument("--num_filters", type=int, default=16, help="channel of for the final contrastive loss specific layer")
    parser.add_argument("--sm_weight", type=float, default=0.1, help="weight for smoothness loss")
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for regularization term")
    parser.add_argument("--lat_weight", type=float, default=10.0, help="weight for latent loss")
    parser.add_argument("--vae_loss_weight", type=float, default=0.4, help="weight for vae loss")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1, help="weight for contrastive loss")

    # 领域自适应相关参数
    parser.add_argument("--domain_loss_weight", type=float, default=0.1, help="weight for domain adaptation loss")
    parser.add_argument("--lambda_grl_max", type=float, default=1.0, help="maximum lambda for gradient reversal layer")
    parser.add_argument("--num_domains", type=int, default=2, help="number of domains (source=0, target=1)")

    # 数据路径
    parser.add_argument("--source_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/train", help="source domain dataset path")
    parser.add_argument("--target_dataset_path", type=str, default="data/ijmond_data/train", help="target domain dataset path")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="pretrained weights path")
    parser.add_argument("--save_path", type=str, default="./checkpoints_domain_adapt/", help="save path for checkpoints")

    return parser.parse_args()


def structure_loss(pred, mask):
    """计算结构损失（二值交叉熵 + IOU损失）"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def compute_segmentation_losses(pred_post_init, pred_post_ref, pred_prior_init, pred_prior_ref, gts, smooth_loss_fn):
    """计算分割相关的损失"""
    # 结构损失
    sal_loss_init_post = structure_loss(pred_post_init, gts)
    sal_loss_ref_post = structure_loss(pred_post_ref, gts)
    sal_loss_init_prior = structure_loss(pred_prior_init, gts)
    sal_loss_ref_prior = structure_loss(pred_prior_ref, gts)

    sal_loss = sal_loss_init_post + sal_loss_ref_post + sal_loss_init_prior + sal_loss_ref_prior

    # 平滑损失 - 使用实例化的损失函数
    sm_loss_post = smooth_loss_fn(torch.sigmoid(pred_post_ref), gts)
    sm_loss_prior = smooth_loss_fn(torch.sigmoid(pred_prior_ref), gts)
    sm_loss = sm_loss_post + sm_loss_prior

    return sal_loss, sm_loss


def train_domain_adaptive_model(opt):
    """领域自适应训练主函数"""
    print(f"开始领域自适应训练，源域: {opt.source_dataset_path}, 目标域: {opt.target_dataset_path}")

    # 创建保存目录
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载器
    print("加载数据...")

    # 构建源域数据路径
    source_image_root = os.path.join(opt.source_dataset_path, "img/")
    source_gt_root = os.path.join(opt.source_dataset_path, "gt/")
    source_trans_root = os.path.join(opt.source_dataset_path, "trans/")

    # 构建目标域数据路径
    target_image_root = os.path.join(opt.target_dataset_path, "img/")
    target_gt_root = os.path.join(opt.target_dataset_path, "gt/")
    target_trans_root = os.path.join(opt.target_dataset_path, "trans/")

    source_loader = get_loader(source_image_root, source_gt_root, source_trans_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=4)
    target_loader = get_loader(target_image_root, target_gt_root, target_trans_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=4)

    print(f"源域数据: {len(source_loader)} batches")
    print(f"目标域数据: {len(target_loader)} batches")

    # 模型构建
    print("构建模型...")
    base_generator = Generator(channel=opt.gen_reduced_channel, latent_dim=opt.latent_dim, num_filters=opt.num_filters)

    # 创建领域自适应模型
    model = create_domain_adaptive_model(
        base_generator=base_generator, feat_channels=opt.feat_channel, num_domains=opt.num_domains, domain_loss_weight=opt.domain_loss_weight
    )

    model.to(device)

    # 加载预训练权重
    if opt.pretrained_weights and os.path.exists(opt.pretrained_weights):
        print(f"加载预训练权重: {opt.pretrained_weights}")
        checkpoint = torch.load(opt.pretrained_weights)
        # 只加载base_generator的权重
        if "generator_state_dict" in checkpoint:
            model.base_generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            model.base_generator.load_state_dict(checkpoint)
        print("预训练权重加载完成")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_gen, betas=(opt.beta, 0.999))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=opt.lr_gen * 0.01)

    # LSC损失设置
    loss_lsc = LocalSaliencyCoherence().cuda()
    loss_lsc_kernels_desc_defaults = [{"weight": 0.1, "xy": 3, "trans": 0.1}]
    loss_lsc_radius = 2
    weight_lsc = 0.01

    # 平滑损失设置
    smooth_loss_fn = smoothness.smoothness_loss(size_average=True).cuda()

    # 训练记录
    loss_record = AvgMeter()

    print("开始训练...")
    for epoch in range(1, opt.epoch + 1):
        model.train()
        loss_record.reset()

        # 计算梯度反转层的lambda值（逐渐增加）
        p = float(epoch - 1) / opt.epoch
        lambda_grl = opt.lambda_grl_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1)

        # 创建目标域数据的循环迭代器
        target_iter = iter(target_loader)

        for i, source_batch in enumerate(source_loader):
            # 获取目标域数据
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            # 源域数据
            src_images, src_gts, src_trans = source_batch
            src_images = src_images.to(device)
            src_gts = src_gts.to(device)
            src_trans = src_trans.to(device)

            # 目标域数据（有标签但在域适应中当作无标签处理）
            tgt_images, tgt_gts, tgt_trans = target_batch
            tgt_images = tgt_images.to(device)
            tgt_gts = tgt_gts.to(device)  # 保留标签但在域适应中不用于监督
            tgt_trans = tgt_trans.to(device)

            optimizer.zero_grad()

            # 源域前向传播
            src_outputs = model(src_images, src_gts, training=True, lambda_grl=lambda_grl, source_domain=True)
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

            # 目标域前向传播（使用标签维持VAE结构但不用于监督）
            # 注意：虽然我们传递了tgt_gts，但在域适应训练中
            # 目标域数据只用于域判别，不用于分割损失计算
            tgt_outputs = model(tgt_images, tgt_gts, training=True, lambda_grl=lambda_grl, source_domain=False)
            (_, _, _, _, _, _, _, d_smoke_tgt, d_bg_tgt) = tgt_outputs

            # 计算各种损失
            # 重要：分割损失只使用源域数据！

            # 1. 源域分割损失（仅源域）
            sal_loss, sm_loss = compute_segmentation_losses(
                sal_init_post_src, sal_ref_post_src, sal_init_prior_src, sal_ref_prior_src, src_gts, smooth_loss_fn
            )

            # 2. 正则化损失
            reg_loss = l2_regularisation(model.base_generator) + l2_regularisation(model)

            # 3. 潜在空间损失
            latent_loss = opt.lat_weight * latent_loss_src

            # 4. 平滑损失
            smooth_loss = opt.sm_weight * sm_loss

            # 5. 对比损失（如果使用自监督学习）
            contrastive_loss = torch.tensor(0.0, device=device)
            if output_post_src is not None:
                contrastive_loss = opt.contrastive_loss_weight * intra_inter_contrastive_loss(
                    output_post_src, src_gts, num_samples=100, margin=1.0, inter=True
                )

            # 6. 域判别损失
            domain_loss, domain_loss_dict = compute_domain_loss(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, src_images.size(0))
            domain_loss = opt.domain_loss_weight * domain_loss

            # LSC损失（后验一致性）
            # 需要缩放图像以匹配LSC要求
            sal_init_post_scale = F.interpolate(sal_init_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            sal_ref_post_scale = F.interpolate(sal_ref_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            sal_init_prior_scale = F.interpolate(sal_init_prior_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            sal_ref_prior_scale = F.interpolate(sal_ref_prior_src, scale_factor=0.3, mode="bilinear", align_corners=True)

            # 创建用于LSC的变换样本
            trans_scale = F.interpolate(src_trans, scale_factor=0.3, mode="bilinear", align_corners=True)
            sample = {"trans": trans_scale}

            # 计算LSC损失
            loss_lsc_1 = loss_lsc(
                torch.sigmoid(sal_init_post_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]

            loss_lsc_2 = loss_lsc(
                torch.sigmoid(sal_ref_post_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]

            loss_lsc_3 = loss_lsc(
                torch.sigmoid(sal_init_prior_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]

            loss_lsc_4 = loss_lsc(
                torch.sigmoid(sal_ref_prior_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]

            loss_lsc_post = weight_lsc * (loss_lsc_1 + loss_lsc_2)
            loss_lsc_prior = weight_lsc * (loss_lsc_3 + loss_lsc_4)

            # 总损失计算
            # VAE损失部分
            gen_loss_cvae = sal_loss + latent_loss + loss_lsc_post
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            # 结构损失部分
            gen_loss_gsnn = 0.5 * (structure_loss(sal_init_prior_src, src_gts) + structure_loss(sal_ref_post_src, src_gts))
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior

            # 总损失
            total_loss = gen_loss_cvae + gen_loss_gsnn + opt.reg_weight * reg_loss + smooth_loss + contrastive_loss + domain_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()

            loss_record.update(total_loss.item(), opt.batchsize)

            # 打印训练信息
            if i % 10 == 0:
                print(
                    f"Epoch [{epoch}/{opt.epoch}], Step [{i}/{len(source_loader)}], "
                    f"Total Loss: {total_loss.item():.4f}, "
                    f"Seg Loss: {sal_loss.item():.4f}, "
                    f"Domain Loss: {domain_loss.item():.4f}, "
                    f"Lambda GRL: {lambda_grl:.3f}, "
                    f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
                )

                # 详细的域损失信息
                print(
                    f'    Domain losses - Smoke Src: {domain_loss_dict["smoke_src"]:.4f}, '
                    f'BG Src: {domain_loss_dict["bg_src"]:.4f}, '
                    f'Smoke Tgt: {domain_loss_dict["smoke_tgt"]:.4f}, '
                    f'BG Tgt: {domain_loss_dict["bg_tgt"]:.4f}'
                )

        # 学习率调度
        scheduler.step()

        # 打印epoch总结
        print(f"Epoch [{epoch}/{opt.epoch}] 完成, 平均损失: {loss_record.avg:.4f}")

        # 保存检查点
        if epoch % 5 == 0 or epoch == opt.epoch:
            checkpoint_path = os.path.join(opt.save_path, f"domain_adapt_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "base_generator_state_dict": model.base_generator.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_record.avg,
                    "lambda_grl": lambda_grl,
                    "opt": opt,
                },
                checkpoint_path,
            )
            print(f"检查点已保存: {checkpoint_path}")

    print("训练完成！")


def main():
    opt = argparser()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("=" * 50)
    print("Trans_BVM 领域自适应训练")
    print("=" * 50)
    print(f"训练参数:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 50)

    train_domain_adaptive_model(opt)


if __name__ == "__main__":
    main()
