#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Domain Adaptive Training Script
Integrates semi-supervised learning and domain adaptation functionality
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

    # ================================== Basic Training Configuration ==================================
    parser.add_argument("--epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=4, help="batch size for training")
    parser.add_argument("--trainsize", type=int, default=352, help="input image resolution (trainsize x trainsize)")

    # ================================== Optimizer Configuration ==================================
    parser.add_argument("--lr_gen", type=float, default=5e-5, help="learning rate for generator")
    parser.add_argument("--beta", type=float, default=0.5, help="beta parameter for Adam optimizer")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping threshold")
    parser.add_argument("--decay_rate", type=float, default=0.8, help="learning rate decay factor for ReduceLROnPlateau")
    parser.add_argument("--decay_epoch", type=int, default=12, help="patience epochs for ReduceLROnPlateau scheduler")

    # ================================== Model Architecture Configuration ==================================
    parser.add_argument(
        "--feat_channel", type=int, default=32, help="feature channel count for saliency features (default: 32, lower for less memory)"
    )
    parser.add_argument("--latent_dim", type=int, default=8, help="latent space dimension (default: 4, lower for less memory)")
    parser.add_argument("--num_filters", type=int, default=8, help="number of filters for contrastive loss layer (default: 8, lower for less memory)")

    # ================================== Loss Function Weight Configuration ==================================
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for L2 regularization")
    parser.add_argument("--lat_weight", type=float, default=2.0, help="weight for latent loss")
    parser.add_argument("--vae_loss_weight", type=float, default=0.6, help="weight for VAE loss component")
    parser.add_argument("--contrastive_loss_weight", type=float, default=1, help="weight for contrastive loss")

    # ================================== Semi-Supervised Learning Configuration ==================================
    parser.add_argument("--inter", action="store_true", default=False, help="use inter-image pixel matching (vs intra-image)")
    parser.add_argument("--no_samples", type=int, default=500, help="number of pixels for contrastive loss sampling")

    # ================================== Domain Adaptation Configuration ==================================
    parser.add_argument("--domain_loss_weight", type=float, default=0.5, help="weight for domain adaptation loss")
    parser.add_argument("--lambda_grl_max", type=float, default=1.0, help="maximum lambda for gradient reversal layer")
    parser.add_argument("--num_domains", type=int, default=2, help="number of domains (source=0, target=1)")
    parser.add_argument("--use_ldconv", action="store_true", default=False, help="use LDConv in domain discriminators (default: False, saves memory)")
    parser.add_argument(
        "--use_attention_pool", action="store_true", default=False, help="use AttentionPool2d in domain discriminators (default: False, saves memory)"
    )

    # ================================== Pseudo Label Learning Configuration ==================================
    parser.add_argument("--pseudo_loss_weight", type=float, default=0.5, help="weight for pseudo label supervision loss")

    # ================================== Dataset Path Configuration ==================================
    parser.add_argument(
        "--source_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/train", help="source domain dataset path (with ground truth labels)"
    )
    parser.add_argument("--target_dataset_path", type=str, default="data/ijmond_data/test", help="target domain dataset path (with pseudo labels)")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="path to pretrained model weights")
    parser.add_argument("--save_model_path", type=str, default="models/domain_adapt", help="directory to save trained models")

    # ================================== Validation and Early Stopping Configuration ==================================
    parser.add_argument("--val_split", type=float, default=0.2, help="fraction of target data used for validation (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=25, help="early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.0001, help="minimum improvement threshold for early stopping")
    parser.add_argument("--enable_validation", action="store_true", default=True, help="enable validation on target data subset")

    # ================================== Data Augmentation and Reproducibility Configuration ==================================
    parser.add_argument("--aug", action="store_true", default=True, help="enable data augmentation for both source and target domain data")
    parser.add_argument("--freeze", action="store_true", default=False, help="freeze randomness for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducible results")

    # ================================== Training Tricks Configuration ==================================
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
        help="number of steps to accumulate gradients before optimizer step (default: 1, >1 to save memory)",
    )

    return parser.parse_args()


def structure_loss(pred, mask):
    """
    Structure loss for evaluating the difference between predicted and ground truth saliency maps.
    Implemented by computing weighted binary cross-entropy loss and weighted IoU loss.
    Args:
        pred: predicted saliency map
        mask: ground truth saliency map
    Returns:
        loss: structure loss
    """
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )  # Calculate weighting factor: larger difference between mask and its local average means higher weight, focusing more on edge or transition areas
    weighted_bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    weighted_bce_loss = (weight * weighted_bce_loss).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))  # Intersection
    union = (((pred + mask - pred * mask)) * weight).sum(dim=(2, 3))  # Union
    weighted_IoU = (inter + 1e-6) / (union + 1e-6)  # Add 1e-6 to prevent division by zero
    weighted_IoU_loss = 1 - weighted_IoU  # IoU loss: higher IoU means lower loss
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
    Load dataset
    Args:
        dataset_path: Dataset path
        opt: Training options
        aug: Whether to enable data augmentation
        freeze: Whether to freeze randomness
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
    Load labeled dataset and perform train/validation split (used when validation is enabled)
    Args:
        dataset_path: Labeled dataset path
        opt: Training options
        freeze: Whether to freeze randomness
    Returns:
        tuple: (train_loader, val_loader, train_step, val_step)
    """
    image_root = os.path.join(dataset_path, "img/")
    gt_root = os.path.join(dataset_path, "gt/")
    trans_map_root = os.path.join(dataset_path, "trans/")

    # Use train/validation split data loaders
    train_loader, val_loader = get_train_val_loaders(
        image_root,
        gt_root,
        trans_map_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        val_split=opt.val_split,
        aug=opt.aug,  # Enable source domain data augmentation to improve generalization
        freeze=freeze,
        random_seed=opt.random_seed,
    )

    train_step = len(train_loader)
    val_step = len(val_loader)
    return train_loader, val_loader, train_step, val_step


def print_training_configuration(opt, device, model_name):
    """
    Print training configuration information
    """
    print("=" * 80)
    print("DOMAIN ADAPTIVE TRAINING CONFIGURATION")
    print("=" * 80)

    # ================================== Basic Configuration ==================================
    print(" BASIC TRAINING SETTINGS")
    print("-" * 40)
    print(f"  Training Epochs: {opt.epoch}")
    print(f"  Batch Size: {opt.batchsize}")
    print(f"  Training Image Size: {opt.trainsize}x{opt.trainsize}")
    print(f"  Device: {device}")
    print(f"  Model Name: {model_name}")

    # ================================== Optimizer Configuration ==================================
    print("\n️  OPTIMIZER SETTINGS")
    print("-" * 40)
    print(f"  Learning Rate: {opt.lr_gen}")
    print(f"  Adam Beta: {opt.beta}")
    print(f"  Gradient Clipping: {opt.clip}")
    print(f"  LR Decay Factor: {opt.decay_rate}")
    print(f"  LR Patience (epochs): {opt.decay_epoch}")

    # ================================== Model Architecture Configuration ==================================
    print("\n️  MODEL ARCHITECTURE")
    print("-" * 40)
    print(f"  Feature Channels: {opt.feat_channel}")
    print(f"  Latent Dimension: {opt.latent_dim}")
    print(f"  Contrastive Layer Filters: {opt.num_filters}")

    # ================================== Loss Function Weights ==================================
    print("\n LOSS FUNCTION WEIGHTS")
    print("-" * 40)
    print(f"  L2 Regularization: {opt.reg_weight}")
    print(f"  Latent Loss: {opt.lat_weight}")
    print(f"  VAE Loss: {opt.vae_loss_weight}")
    print(f"  Contrastive Loss: {opt.contrastive_loss_weight}")
    print(f"  Domain Adaptation Loss: {opt.domain_loss_weight}")
    print(f"  Pseudo Label Loss: {opt.pseudo_loss_weight}")

    # ================================== Domain Adaptation Settings ==================================
    print("\n DOMAIN ADAPTATION SETTINGS")
    print("-" * 40)
    print(f"  Number of Domains: {opt.num_domains}")
    print(f"  Gradient Reversal Lambda Max: {opt.lambda_grl_max}")
    print(f"  Use LDConv in Discriminators: {opt.use_ldconv}")
    print(f"  Use AttentionPool2d in Discriminators: {opt.use_attention_pool}")
    print(f"  Pseudo Label Weight: {opt.pseudo_loss_weight}")

    # ================================== Semi-Supervised Learning ==================================
    print("\n WEAKLY-SUPERVISED LEARNING")
    print("-" * 40)
    print(f"  Contrastive Pixel Matching: {'Inter-image' if opt.inter else 'Intra-image'}")
    print(f"  Contrastive Sample Count: {opt.no_samples}")

    # ================================== Dataset Configuration ==================================
    print("\n DATASET CONFIGURATION")
    print("-" * 40)
    print(f"  Source Domain Path: {opt.source_dataset_path}")
    print(f"  Target Domain Path: {opt.target_dataset_path}")
    print(f"  Pretrained Weights: {opt.pretrained_weights or 'None'}")
    print(f"  Model Save Path: {opt.save_model_path}")

    # ================================== Validation and Early Stopping ==================================
    print("\n VALIDATION & EARLY STOPPING")
    print("-" * 40)
    print(f"  Enable Validation: {opt.enable_validation}")
    print(f"  Validation Domain: Target Domain")
    print(f"  Validation Split: {opt.val_split} (of target data)")
    print(f"  Early Stopping Patience: {opt.patience}")
    print(f"  Min Delta for Improvement: {opt.min_delta}")
    if opt.enable_validation:
        print("   Validation Strategy: Target domain split for training/validation")
        print("     - Source domain: Full dataset for training")
        print("     - Target domain: Split into training/validation sets")

    # ================================== Data Augmentation & Reproducibility ==================================
    print("\n DATA AUGMENTATION & REPRODUCIBILITY")
    print("-" * 40)
    print(f"  Data Augmentation (Both Domains): {opt.aug}")
    print(f"  Freeze Randomness: {opt.freeze}")
    print(f"  Random Seed: {opt.random_seed}")
    if opt.freeze and opt.aug:
        print("  NOTE: Data augmentation disabled due to freeze mode")

    print("=" * 80)


opt = argparser()

# Set random seed
torch.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.random_seed)

# Get dataset name and generate model name
source_dataset_name = get_dataset_name_from_path(opt.source_dataset_path)
target_dataset_name = get_dataset_name_from_path(opt.target_dataset_path)

# Use domain adaptation-specific model naming function
model_name = generate_domain_adaptation_model_name(source_dataset_name, target_dataset_name, opt.pretrained_weights)

original_save_path = opt.save_model_path
opt.save_model_path = os.path.join(original_save_path, model_name)

# Print training configuration
print_training_configuration(opt, device, model_name)


# Data loaders
print("\n LOADING DATASETS...")


# Load source domain data (always use all source data for training in domain adaptation)
source_train_loader, source_train_step = load_data(opt.source_dataset_path, opt, aug=opt.aug, freeze=opt.freeze)
print(f"Source domain training set: {source_train_step} batches (full dataset)")


# Load target domain data (with or without validation split)
if opt.enable_validation:
    # Validation mode enabled: target domain split into training/validation sets
    target_train_loader, val_loader, target_train_step, val_step = load_labeled_data_with_validation(opt.target_dataset_path, opt, freeze=opt.freeze)
    print(f"Target domain training set: {target_train_step} batches")
    print(f"Target domain validation set: {val_step} batches")

    # Initialize early stopping - unified based on training loss
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_train_loss = float("inf")
    best_epoch = 0
    validation_enabled = True
else:
    # Non-validation mode: use all target domain data for training
    target_train_loader, target_train_step = load_data(opt.target_dataset_path, opt, aug=opt.aug, freeze=opt.freeze)
    val_loader = None
    print(f"Target domain training set: {target_train_step} batches (full dataset)")

    # Initialize early stopping - unified based on training loss
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_train_loss = float("inf")
    best_epoch = 0
    validation_enabled = False


# Create cyclic iterator for target domain data
target_train_iter = cycle(target_train_loader)  # continuously iterate over the target dataset


# Use source data loader for main training loop
train_loader = source_train_loader
total_step = source_train_step


# Model construction
print("Building model...")
base_generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=opt.num_filters)


# Create domain adaptive model
model = create_domain_adaptive_model(
    base_generator=base_generator,
    feat_channels=opt.feat_channel,
    num_domains=opt.num_domains,
    domain_loss_weight=opt.domain_loss_weight,
    use_ldconv=opt.use_ldconv,
    use_attention_pool=opt.use_attention_pool,
)

model.to(device)


# Load pretrained weights
if opt.pretrained_weights and os.path.exists(opt.pretrained_weights):
    print(f"Loading pretrained weights: {opt.pretrained_weights}")
    checkpoint = torch.load(opt.pretrained_weights, map_location=device)
    # Only load base_generator weights
    if "generator_state_dict" in checkpoint:
        model.base_generator.load_state_dict(checkpoint["generator_state_dict"])
    else:
        model.base_generator.load_state_dict(checkpoint)
    print("Pretrained weights loaded")


# Optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr_gen, betas=(opt.beta, 0.999))
# Use ReduceLROnPlateau scheduler, adaptively adjust learning rate based on loss
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # Monitor loss, reduce learning rate when loss stops decreasing
    factor=opt.decay_rate,  # Learning rate decay factor
    patience=opt.decay_epoch,  # Number of epochs to wait before reducing learning rate if no improvement
    min_lr=1e-7,  # Minimum learning rate
)

print(f"Learning Rate Scheduler configured:")
print(f"  - Type: ReduceLROnPlateau (adaptive based on loss)")
print(f"  - Patience (epochs to wait): {opt.decay_epoch}")
print(f"  - Decay Factor: {opt.decay_rate}")
print(f"  - Minimum LR: 1e-7")


# Loss functions
size_rates = [1]  # multi-scale training
loss_lsc = LocalSaliencyCoherence().to(device)  # Local saliency coherence loss function
loss_lsc_kernels_desc_defaults = [{"weight": 0.1, "xy": 3, "trans": 0.1}]
loss_lsc_radius = 2
weight_lsc = 0.01


print("Let's go!")
# Ensure save directory exists before training starts
save_path = opt.save_model_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created save directory: {save_path}")

for epoch in range(1, opt.epoch + 1):
    print("--" * 10 + "Epoch: {}/{}".format(epoch, opt.epoch) + "--" * 10)
    model.train()
    loss_record = AvgMeter()
    print("Learning Rate: {}".format(optimizer.param_groups[0]["lr"]))

    # Compute lambda for gradient reversal layer (gradually increase)
    p = float(epoch - 1) / opt.epoch
    lambda_grl = opt.lambda_grl_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1)

    for i, source_pack in enumerate(train_loader, start=1):
        # Load a batch from the target loader
        target_pack = next(target_train_iter)

        for rate in size_rates:
            # Gradient accumulation: only zero_grad at the first rate
            if (i - 1) % opt.accumulation_steps == 0:
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
            gts_tgt = gts_tgt.to(device)  # Target domain pseudo labels, used for training
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

            ### Source domain forward pass ############################
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

            ### Target domain forward pass ############################
            # Target domain data used for domain adaptation and pseudo label supervised learning
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

            ### Loss calculation ############################

            # 1. Source domain supervised loss (structure loss)
            src_sal_loss = 0.5 * (structure_loss(sal_init_post_src, gts_src) + structure_loss(sal_ref_post_src, gts_src))

            # 2. Target domain pseudo label supervised loss
            tgt_sal_loss = 0.5 * (structure_loss(sal_init_post_tgt, gts_tgt) + structure_loss(sal_ref_post_tgt, gts_tgt))

            # Total segmentation loss
            sal_loss = src_sal_loss + opt.pseudo_loss_weight * tgt_sal_loss

            # 3. Contrastive loss (source and target domain)
            cont_loss_src = intra_inter_contrastive_loss(output_post_src, gts_src, num_samples=opt.no_samples, margin=1.0, inter=opt.inter)
            cont_loss_tgt = intra_inter_contrastive_loss(output_post_tgt, gts_tgt, num_samples=opt.no_samples, margin=1.0, inter=opt.inter)
            cont_loss = cont_loss_src + opt.pseudo_loss_weight * cont_loss_tgt

            # 4. LSC loss calculation (source + target domain)
            # Source domain LSC loss
            trans_scale_src = F.interpolate(trans_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_init_scale_src = F.interpolate(sal_init_prior_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_ref_scale_src = F.interpolate(sal_ref_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_init_scale_src = F.interpolate(sal_init_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_ref_scale_src = F.interpolate(sal_ref_post_src, scale_factor=0.3, mode="bilinear", align_corners=True)
            sample_src = {"trans": trans_scale_src}

            # Source domain LSC loss calculation
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

            # Target domain LSC loss
            trans_scale_tgt = F.interpolate(trans_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_init_scale_tgt = F.interpolate(sal_init_prior_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_ref_scale_tgt = F.interpolate(sal_ref_post_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_init_scale_tgt = F.interpolate(sal_init_post_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_ref_scale_tgt = F.interpolate(sal_ref_post_tgt, scale_factor=0.3, mode="bilinear", align_corners=True)
            sample_tgt = {"trans": trans_scale_tgt}

            # Target domain LSC loss calculation
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

            # Total LSC loss
            loss_lsc_post = loss_lsc_post_src + opt.pseudo_loss_weight * loss_lsc_post_tgt
            loss_lsc_prior = loss_lsc_prior_src + opt.pseudo_loss_weight * loss_lsc_prior_tgt

            # 4. L2 regularization loss
            reg_loss = (
                l2_regularisation(model.base_generator.xy_encoder)
                + l2_regularisation(model.base_generator.x_encoder)
                + l2_regularisation(model.base_generator.sal_encoder)
            )
            reg_loss = opt.reg_weight * reg_loss

            # 5. Latent loss (linear annealing)
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            latent_loss = opt.lat_weight * anneal_reg * (latent_loss_src + latent_loss_tgt)

            # 6. Domain discrimination loss
            domain_loss, domain_loss_dict = compute_domain_loss(d_smoke_src, d_bg_src, d_smoke_tgt, d_bg_tgt, images_src.size(0))
            domain_loss = opt.domain_loss_weight * domain_loss

            # VAE loss part (includes source and target domain)
            gen_loss_cvae = sal_loss + latent_loss + loss_lsc_post
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            # Structure loss part (includes source and target domain)
            gen_loss_gsnn = 0.5 * (
                structure_loss(sal_init_prior_src, gts_src)
                + structure_loss(sal_ref_post_src, gts_src)
                + opt.pseudo_loss_weight * (structure_loss(sal_init_prior_tgt, gts_tgt) + structure_loss(sal_ref_post_tgt, gts_tgt))
            )
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior

            ### Total loss ###############################################
            total_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss + domain_loss + opt.contrastive_loss_weight * cont_loss  # type: torch.Tensor
            total_loss = total_loss / opt.accumulation_steps  # Gradient scaling
            total_loss.backward()

            # Gradient clipping
            if opt.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

            # Only step and clear cache after accumulating to specified steps
            if (i % opt.accumulation_steps == 0) or (i == total_step):
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if rate == 1:
                loss_record.update(total_loss.data * opt.accumulation_steps, opt.batchsize)

        # Print training info - print at 25%, 50%, 75%, 100%
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

            print(log_info)  # Call scheduler.step() after training loop ends - ReduceLROnPlateau requires the monitored metric
    old_lr = optimizer.param_groups[0]["lr"]

    # Choose monitoring metric based on whether validation is enabled
    if validation_enabled and val_loader is not None:
        # If validation is enabled, call scheduler.step(val_loss) after validation
        pass
    else:
        # If validation is not enabled, use training loss
        current_loss = loss_record.avg
        # scheduler.step(current_loss)  # Note: originally used training loss to adjust learning rate

    current_lr = optimizer.param_groups[0]["lr"]

    if old_lr != current_lr:
        print(f"Epoch {epoch} completed. Learning rate changed: {old_lr:.6f} -> {current_lr:.6f}")
    else:
        print(f"Epoch {epoch} completed. Learning rate: {current_lr:.6f}")

    # Validation and early stopping logic - unified based on validation loss (val_loss)
    # current_train_loss = loss_record.avg  # Note: originally used training loss for early stopping

    if validation_enabled and val_loader is not None:
        # Validation mode enabled: evaluate model on target domain validation set
        print("Starting validation on target domain...")
        with torch.no_grad():
            val_loss, val_metrics = validate_model(model.base_generator, val_loader, device, structure_loss)

        print(f"Target Domain Validation Results (Reference Only) - Loss: {val_loss:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}")
        print(f"  F1-Score: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

        # Update learning rate scheduler using validation loss
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Learning rate adjusted based on validation loss: {current_lr:.6f} -> {new_lr:.6f}")

        # Unified model saving and early stopping logic - based on validation loss
        current_val_loss = val_loss
        if current_val_loss < best_train_loss:
            best_train_loss = current_val_loss
            best_epoch = epoch
            # Save best model
            best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
            torch.save(model.base_generator.state_dict(), os.path.join(opt.save_model_path, best_model_filename))
            print(f"New best model saved! Validation loss: {current_val_loss:.4f}")
            print(f"  Corresponding target validation IoU: {val_metrics['iou']:.4f}")

        # ========== New: save checkpoint every 5 epochs ===========
        if epoch % 5 == 0:
            checkpoint_filename = f"checkpoint_epoch_{epoch:03d}.pth"
            torch.save(model.base_generator.state_dict(), os.path.join(opt.save_model_path, checkpoint_filename))
            print(f"Checkpoint saved at epoch {epoch}: {checkpoint_filename}")
        # ======================================================

        early_stopping(current_val_loss, model.base_generator)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation loss: {best_train_loss:.4f} achieved at epoch {best_epoch}")
            break
    else:
        # If validation is not enabled, use training loss
        current_train_loss = loss_record.avg
        # if current_train_loss < best_train_loss:
        #     best_train_loss = current_train_loss
        #     best_epoch = epoch
        #     best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
        #     torch.save(model.base_generator.state_dict(), os.path.join(opt.save_model_path, best_model_filename))
        #     print(f"New best model saved! Training loss: {current_train_loss:.4f}")
        # early_stopping(current_train_loss, model.base_generator)
        # if early_stopping.early_stop:
        #     print(f"Early stopping triggered at epoch {epoch}")
        #     print(f"Best training loss: {best_train_loss:.4f} achieved at epoch {best_epoch}")
        #     break

# Summary after training ends
print("\n" + "=" * 50)
print("Two-Domain Adaptive Training with Pseudo Labels completed!")
print(f"Source Domain: {source_dataset_name} (Ground Truth Labels - Full Dataset)")
print(f"Target Domain: {target_dataset_name} (Pseudo Labels)")
print(f"Pseudo Label Weight: {opt.pseudo_loss_weight}")
print(f"Best training loss: {best_train_loss:.4f} achieved at epoch {best_epoch}")
best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
print(f"Best model saved at: {os.path.join(opt.save_model_path, best_model_filename)}")
print("=" * 50)
