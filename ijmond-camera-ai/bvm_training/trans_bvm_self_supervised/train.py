import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from dataloader import get_loader, get_dataset_name_from_path, get_train_val_loaders
from utils import adjust_lr, AvgMeter, EarlyStopping, validate_model, generate_model_name, generate_checkpoint_filename, generate_best_model_filename
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
import smoothness
from lscloss import *
from itertools import cycle
from cont_loss import intra_inter_contrastive_loss
from PIL import Image

# Define computation device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def argparser():
    parser = argparse.ArgumentParser()

    # ================== 基础训练参数 ==================
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")  # 训练轮数
    parser.add_argument("--batchsize", type=int, default=6, help="number of samples per batch")  # 批量大小
    parser.add_argument(
        "--trainsize", type=int, default=352, help="input image resolution (trainsize x trainsize)"
    )  # 输入图像分辨率，训练时的图像大小，别随便调！！

    # ================== 优化器参数 ==================
    parser.add_argument("--lr_gen", type=float, default=2.5e-5, help="learning rate for generator")  # 生成器学习率
    parser.add_argument("--lr_des", type=float, default=2.5e-5, help="learning rate for descriptor")  # 描述器学习率
    parser.add_argument("--beta", type=float, default=0.5, help="beta of Adam for generator")  # Adam优化器的beta参数
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping margin")  # 梯度裁剪边际，用于防止梯度爆炸
    parser.add_argument("--decay_rate", type=float, default=0.9, help="decay rate of learning rate")  # 学习率衰减率，用于调整学习率
    parser.add_argument("--decay_epoch", type=int, default=20, help="every n epochs decay learning rate")  # 学习率衰减周期

    # ================== 模型架构参数 ==================
    parser.add_argument("--gen_reduced_channel", type=int, default=32, help="reduced channel in generator")  # 生成器中减少的通道数
    parser.add_argument("--des_reduced_channel", type=int, default=64, help="reduced channel in descriptor")  # 描述器中减少的通道数
    parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")  # 重要性特征的通道数
    parser.add_argument("--latent_dim", type=int, default=3, help="latent dim")  # 潜在维度，用于生成器和描述器的潜在空间
    parser.add_argument(
        "--num_filters", type=int, default=16, help="channel of for the final contrastive loss specific layer"
    )  # 最终对比损失特定层的通道数

    # ================== EBM相关参数 ==================
    parser.add_argument(
        "--langevin_step_num_des", type=int, default=10, help="number of langevin steps for ebm"
    )  # EBM的langevin步骤数，EBM是能量基模型，langevin步骤是指在生成过程中使用的迭代步骤数
    parser.add_argument("--langevin_step_size_des", type=float, default=0.026, help="step size of EBM langevin")  # EBM langevin的步长
    parser.add_argument(
        "--energy_form", default="identity", help="tanh | sigmoid | identity | softplus"
    )  # 能量函数的形式，tanh：双曲正切函数，sigmoid：S型函数，identity：恒等函数，softplus：平滑的ReLU函数

    # ================== 损失函数权重 ==================
    parser.add_argument("--sm_weight", type=float, default=0.1, help="weight for smoothness loss")  # 平滑性损失的权重
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for regularization term")  # 正则化项的权重
    parser.add_argument("--lat_weight", type=float, default=10.0, help="weight for latent loss")  # 潜在损失的权重
    parser.add_argument("--vae_loss_weight", type=float, default=0.4, help="weight for vae loss")  # VAE损失的权重，VAE是变分自编码器，用于生成模型
    parser.add_argument(
        "--contrastive_loss_weight", type=float, default=0.1, help="weight for contrastive loss"
    )  # 对比损失的权重，对比损失用于增强模型对不同样本间的区分能力

    # ================== 半监督学习特有参数 ==================
    parser.add_argument(
        "--inter", action="store_true", default=False, help="Inter pixel (different image) match if True, else intra pixel (same image) match."
    )  # 是否进行跨图像像素匹配，如果为True，则在不同图像之间进行匹配，否则在同一图像内进行匹配
    parser.add_argument("--no_samples", type=int, default=50, help="number of pixels to consider in the contrastive loss")  # 对比损失中考虑的像素数量

    # ================== 数据集路径 ==================
    parser.add_argument(
        "--labeled_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/train", help="labeled dataset path"
    )  # 标注数据集路径
    parser.add_argument(
        "--unlabeled_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/weak_supervision", help="unlabeled dataset path"
    )  # 无标注数据集路径
    parser.add_argument("--pretrained_weights", type=str, default=None, help="pretrained weights. it can be none")  # 预训练权重路径，可以为None
    parser.add_argument("--save_model_path", type=str, default="models/self-supervision", help="model save path")  # 模型保存路径

    # ================== 校验和早停参数 ==================
    parser.add_argument("--val_split", type=float, default=0.2, help="fraction of labeled dataset used for validation (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience")  # 早停耐心值
    parser.add_argument("--min_delta", type=float, default=0.001, help="minimum improvement for early stopping")  # 早停最小改善值
    parser.add_argument("--enable_validation", action="store_true", default=True, help="enable validation on labeled data subset")  # 启用校验

    # ================== 数据增强和可重现性参数 ==================
    parser.add_argument(
        "--aug", action="store_true", default=False, help="enable data augmentation for unlabeled data"
    )  # 启用数据增强（仅对无标注数据）
    parser.add_argument("--freeze", action="store_true", default=False, help="freeze all randomness for full reproducibility")  # 冻结所有随机性
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")  # 随机种子

    opt = parser.parse_args()
    return opt


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

    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
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


def save_tensor_as_image(tensor, path):
    # Clamp values to ensure they are within a valid range
    tensor_clamped = torch.clamp(tensor, 0, 1)
    # Scale values from 0-1 to 0-255
    tensor_scaled = (tensor_clamped * 255).byte()
    print(np.unique(tensor_scaled))
    # # Convert to numpy
    tensor_np = tensor_scaled.cpu().numpy()
    # # Handle different dimensions (C, H, W) vs (H, W)
    if tensor_np.ndim == 3:  # If tensor is in CHW format
        tensor_np = tensor_np.transpose(1, 2, 0)  # Convert CHW to HWC for image
    print(tensor_np.dtype)
    # # Convert to Image
    cv2.imwrite("path.png", tensor_np)
    # img = Image.fromarray(tensor_np)
    # # Save the image
    # img.save(path)
    # print(f'Image saved to {path}')


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
    加载数据集（半监督学习专用）
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


opt = argparser()

# 获取数据集名称并生成模型名称
labeled_dataset_name = get_dataset_name_from_path(opt.labeled_dataset_path)
unlabeled_dataset_name = get_dataset_name_from_path(opt.unlabeled_dataset_path)
model_name = generate_model_name(labeled_dataset_name, unlabeled_dataset_name, opt.pretrained_weights)
original_save_path = opt.save_model_path
opt.save_model_path = os.path.join(original_save_path, model_name)

# 打印训练配置信息
print("\n========== Semi-Supervised Training Configuration ==========")
print("Training Epochs: {}".format(opt.epoch))
print("Learning Rates:")
print("  - Generator: {}".format(opt.lr_gen))
print("  - Descriptor: {}".format(opt.lr_des))
print("\nOptimization Settings:")
print("  - Batch Size: {}".format(opt.batchsize))
print("  - Training Size: {}".format(opt.trainsize))
print("  - Gradient Clip: {}".format(opt.clip))
print("  - Adam Beta: {}".format(opt.beta))
print("\nLearning Rate Schedule:")
print("  - Decay Rate: {}".format(opt.decay_rate))
print("  - Decay Epoch: {}".format(opt.decay_epoch))
print("\nModel Architecture:")
print("  - Generator Reduced Channel: {}".format(opt.gen_reduced_channel))
print("  - Descriptor Reduced Channel: {}".format(opt.des_reduced_channel))
print("  - Feature Channel: {}".format(opt.feat_channel))
print("  - Latent Dimension: {}".format(opt.latent_dim))
print("  - Num Filters: {}".format(opt.num_filters))
print("\nLoss Weights:")
print("  - Smoothness Weight: {}".format(opt.sm_weight))
print("  - Regularization Weight: {}".format(opt.reg_weight))
print("  - Latent Loss Weight: {}".format(opt.lat_weight))
print("  - VAE Loss Weight: {}".format(opt.vae_loss_weight))
print("  - Contrastive Loss Weight: {}".format(opt.contrastive_loss_weight))
print("\nPaths:")
print("  - Labeled Dataset Path: {}".format(opt.labeled_dataset_path))
print("  - Unlabeled Dataset Path: {}".format(opt.unlabeled_dataset_path))
print("  - Labeled Dataset Name: {}".format(labeled_dataset_name))
print("  - Unlabeled Dataset Name: {}".format(unlabeled_dataset_name))
print("  - Model Name: {}".format(model_name))
print("  - Original Save Path: {}".format(original_save_path))
print("  - Final Save Path: {}".format(opt.save_model_path))
print("  - Pretrained Weights: {}".format(opt.pretrained_weights))
print("\nValidation Settings:")
print("  - Validation Enabled: {}".format("Yes" if opt.enable_validation else "No"))
print("  - Validation Split: {}".format(opt.val_split))
print("  - Early Stopping Patience: {}".format(opt.patience))
print("  - Min Delta: {}".format(opt.min_delta))
print("\nEBM Settings:")
print("  - Langevin Steps: {}".format(opt.langevin_step_num_des))
print("  - Langevin Step Size: {}".format(opt.langevin_step_size_des))
print("  - Energy Form: {}".format(opt.energy_form))
print("\nSemi-Supervised Settings:")
print("  - Inter-pixel matching: {}".format("Enabled" if opt.inter else "Disabled"))
print("  - Contrastive samples: {}".format(opt.no_samples))
print("\nData Augmentation & Reproducibility:")
print("  - Data Augmentation (unlabeled): {}".format("Enabled" if opt.aug else "Disabled"))
print("  - Freeze Mode: {}".format("Enabled" if opt.freeze else "Disabled"))
print("  - Random Seed: {}".format(opt.random_seed))
if opt.freeze:
    print("  - [WARNING] Freeze mode enabled - all randomness frozen for debugging")
if opt.freeze and opt.aug:
    print("  - [INFO] Data augmentation will be disabled due to freeze mode")
print("==========================================\n")

print(f"Using device: {device}")
print("Generator Learning Rate: {}".format(opt.lr_gen))

# Build model
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=opt.num_filters)
if opt.pretrained_weights is not None:
    print(f"Load pretrained weights: {opt.pretrained_weights}")
    generator.load_state_dict(torch.load(opt.pretrained_weights, map_location=device))
generator.to(device)  # 使用统一的设备管理
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=(opt.beta, 0.999))  # 修复betas参数

# Load labeled data (with or without validation split)
if opt.enable_validation:
    # 启用校验模式：使用训练/验证分割
    train_loader_labeled, val_loader, total_step_labeled, val_step = load_labeled_data_with_validation(
        opt.labeled_dataset_path, opt, freeze=opt.freeze
    )
    print(f"Labeled training set size: {total_step_labeled}")
    print(f"Validation set size: {val_step}")

    # 初始化早停策略 - 基于验证指标
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_val_iou = 0.0
    best_epoch = 0
    validation_enabled = True
else:
    # 非校验模式：使用所有标注数据进行训练
    train_loader_labeled, total_step_labeled = load_data(opt.labeled_dataset_path, opt, aug=False, freeze=opt.freeze)
    val_loader = None
    print(f"Labeled dataset size: {total_step_labeled}")

    # 初始化早停策略 - 基于训练损失
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_val_loss = float("inf")
    best_epoch = 0
    validation_enabled = False

# Load pseudo labeled data (with augmentation if enabled)
train_loader_un, total_step_un = load_data(opt.unlabeled_dataset_path, opt, aug=opt.aug, freeze=opt.freeze)
train_loader_un_iter = cycle(train_loader_un)  # continuously iterate over the pseudo-labeled dataset
print(f"Unlabeled dataset size: {total_step_un}")

# Use labeled data loader for main training loop
train_loader = train_loader_labeled
total_step = total_step_labeled

# Loss functions
scheduler = lr_scheduler.StepLR(generator_optimizer, step_size=10, gamma=0.5)
size_rates = [1]  # multi-scale training
smooth_loss = smoothness.smoothness_loss(size_average=True)  # 平滑性损失函数，约束生成的图像平滑性
loss_lsc = LocalSaliencyCoherence().to(device)  # 局部显著性一致性损失函数，在细粒度区域加强预测的一致性
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
    generator.train()
    loss_record = AvgMeter()
    print("Generator Learning Rate: {}".format(generator_optimizer.param_groups[0]["lr"]))
    for i, pack in enumerate(train_loader, start=1):
        # Load a batch from the pseudo-labeled loader
        pseudo_labeled_pack = next(train_loader_un_iter)
        for rate in size_rates:
            generator_optimizer.zero_grad()
            ### Load Data ######################################
            # Unpack labeled data
            images_lb, gts_lb, trans_lb = pack
            num_labeled_data = images_lb.size(0)
            # Use unified device management
            images_lb = images_lb.to(device)
            gts_lb = gts_lb.to(device)
            trans_lb = trans_lb.to(device)
            # Unpack pseudo-labeled data
            images_un, gts_un, trans_un = pseudo_labeled_pack
            images_un = images_un.to(device)
            gts_un = gts_un.to(device)
            trans_un = trans_un.to(device)
            ### Concatenate the labeled and unlabeled samples #####
            images = torch.cat((images_lb, images_un), dim=0)
            gts = torch.cat((gts_lb, gts_un), dim=0)
            trans = torch.cat((trans_lb, trans_un), dim=0)
            ### Feed the network ############################
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                trans = F.interpolate(trans, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
            # forward supports x,y as inputs
            pred_post_init, pred_post_ref, pred_prior_init, pred_piror_ref, latent_loss, out_post, out_prior = generator(images, gts)
            ### Calculate contrastive loss ##################
            cont_loss = intra_inter_contrastive_loss(out_post, gts, num_samples=opt.no_samples, margin=1.0, inter=opt.inter)
            ### Continue only with the labeled data ########################
            # re-scale data for crf loss
            trans_scale = F.interpolate(trans_lb, scale_factor=0.3, mode="bilinear", align_corners=True)
            images_scale = F.interpolate(images_lb, scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_init_scale = F.interpolate(pred_prior_init[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_prior_ref_scale = F.interpolate(pred_post_ref[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_init_scale = F.interpolate(pred_post_init[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
            pred_post_ref_scale = F.interpolate(pred_post_ref[:num_labeled_data], scale_factor=0.3, mode="bilinear", align_corners=True)
            sample = {"trans": trans_scale}

            loss_lsc_1 = loss_lsc(
                torch.sigmoid(pred_post_init_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_2 = loss_lsc(
                torch.sigmoid(pred_post_ref_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_post = weight_lsc * (loss_lsc_1 + loss_lsc_2)
            ## l2 regularizer the inference model
            reg_loss = l2_regularisation(generator.xy_encoder) + l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
            reg_loss = opt.reg_weight * reg_loss
            latent_loss = latent_loss

            sal_loss = 0.5 * (structure_loss(pred_post_init[:num_labeled_data], gts_lb) + structure_loss(pred_post_ref[:num_labeled_data], gts_lb))
            anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
            latent_loss = opt.lat_weight * anneal_reg * latent_loss

            loss_lsc_3 = loss_lsc(
                torch.sigmoid(pred_prior_init_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_4 = loss_lsc(
                torch.sigmoid(pred_prior_ref_scale),
                loss_lsc_kernels_desc_defaults,
                loss_lsc_radius,
                sample,
                trans_scale.shape[2],
                trans_scale.shape[3],
            )["loss"]
            loss_lsc_prior = weight_lsc * (loss_lsc_3 + loss_lsc_4)

            gen_loss_cvae = sal_loss + latent_loss + loss_lsc_post
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            gen_loss_gsnn = 0.5 * (
                structure_loss(pred_prior_init[:num_labeled_data], gts_lb) + structure_loss(pred_post_ref[:num_labeled_data], gts_lb)
            )
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior

            ### Total loss ###############################################
            gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss + opt.contrastive_loss_weight * cont_loss # type: torch.Tensor
            gen_loss.backward()

            # Gradient clipping
            if opt.clip > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), opt.clip)

            generator_optimizer.step()

            if rate == 1:
                loss_record.update(gen_loss.data, opt.batchsize)  # 这里传入tensor而不是标量

        if i % 10 == 0 or i == total_step:
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}".format(
                    datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()
                )
            )

    # 在训练循环结束后调用scheduler.step()
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch} completed. Current learning rate: {current_lr}")

    # 校验和早停逻辑
    if validation_enabled and val_loader is not None:
        # 启用校验模式：在验证集上评估模型
        print("Starting validation...")
        val_loss, val_metrics = validate_model(generator, val_loader, device, structure_loss)

        print(f"Validation Results - Loss: {val_loss:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}")
        print(f"  F1-Score: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

        # 检查是否是最佳模型 - 使用IoU作为主要指标
        current_iou = val_metrics["iou"]
        if current_iou > best_val_iou:
            best_val_iou = current_iou
            best_epoch = epoch
            # 保存最佳模型
            best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
            torch.save(generator.state_dict(), os.path.join(opt.save_model_path, best_model_filename))
            print(f"New best model saved! Validation IoU: {current_iou:.4f}")

        # 早停检查 - 使用IoU
        early_stopping(current_iou, generator)
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
            torch.save(generator.state_dict(), os.path.join(opt.save_model_path, best_model_filename))
            print(f"New best model saved! Training loss: {current_loss:.4f}")

        # 早停检查 - 使用训练损失
        early_stopping(current_loss, generator)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best training loss: {best_val_loss:.4f} achieved at epoch {best_epoch}")
            break

    # 定期保存检查点 - 使用动态文件名
    if epoch >= 0 and epoch % 10 == 0:
        checkpoint_filename = generate_checkpoint_filename(epoch, model_name, opt.pretrained_weights)
        torch.save(generator.state_dict(), os.path.join(opt.save_model_path, checkpoint_filename))
        print(f"Checkpoint saved: {checkpoint_filename}")

# 训练结束后的总结
print("\n" + "=" * 50)
print("Semi-Supervised Training completed!")
if validation_enabled:
    print(f"Best validation IoU: {best_val_iou:.4f} achieved at epoch {best_epoch}")
else:
    print(f"Best training loss: {best_val_loss:.4f} achieved at epoch {best_epoch}")
best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
print(f"Best model saved at: {os.path.join(opt.save_model_path, best_model_filename)}")
print("=" * 50)
