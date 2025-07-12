import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from dataloader import get_loader, get_dataset_name_from_path, get_train_val_loaders
from utils import AvgMeter, EarlyStopping, validate_model, generate_model_name, generate_checkpoint_filename, generate_best_model_filename
from utils import l2_regularisation
from lscloss import *
from itertools import cycle
from cont_loss import intra_inter_contrastive_loss
from PIL import Image

# Define computation device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def argparser():
    parser = argparse.ArgumentParser(description="Weakly-Supervised Training Script")

    # ================================== Basic Training Configuration ==================================
    parser.add_argument("--epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=8, help="batch size for training")
    parser.add_argument("--trainsize", type=int, default=352, help="input image resolution (trainsize x trainsize)")

    # ================================== Optimizer Configuration ==================================
    parser.add_argument("--lr_gen", type=float, default=5e-5, help="learning rate for generator")
    parser.add_argument("--beta", type=float, default=0.5, help="beta parameter for Adam optimizer")
    parser.add_argument("--clip", type=float, default=0.5, help="gradient clipping threshold")
    parser.add_argument("--decay_rate", type=float, default=0.8, help="learning rate decay factor for ReduceLROnPlateau")
    parser.add_argument("--decay_epoch", type=int, default=12, help="patience epochs for ReduceLROnPlateau scheduler")

    # ================================== Model Architecture Configuration ==================================
    parser.add_argument("--feat_channel", type=int, default=32, help="feature channel count for saliency features")
    parser.add_argument("--latent_dim", type=int, default=8, help="latent space dimension")
    parser.add_argument("--num_filters", type=int, default=16, help="number of filters for contrastive loss layer")

    # ================================== Loss Function Weight Configuration ==================================
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for L2 regularization")
    parser.add_argument("--lat_weight", type=float, default=2.0, help="weight for latent loss")
    parser.add_argument("--vae_loss_weight", type=float, default=0.6, help="weight for VAE loss component")  # Reduced from 1.0 to 0.6
    parser.add_argument("--contrastive_loss_weight", type=float, default=1, help="weight for contrastive loss")

    # ================================== WEAKLY-Supervised Learning Configuration ==================================
    parser.add_argument("--inter", action="store_true", default=False, help="use inter-image pixel matching (vs intra-image)")
    parser.add_argument("--no_samples", type=int, default=500, help="number of pixels for contrastive loss sampling")

    # ================================== Dataset Path Configuration ==================================
    parser.add_argument("--labeled_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/train", help="path to labeled dataset")
    parser.add_argument(
        "--unlabeled_dataset_path", type=str, default="data/SMOKE5K_Dataset/SMOKE5K/weak_supervision", help="path to unlabeled dataset"
    )
    parser.add_argument("--pretrained_weights", type=str, default=None, help="path to pretrained model weights")
    parser.add_argument("--save_model_path", type=str, default="models/weak-supervision", help="directory to save trained models")

    # ================================== Validation and Early Stopping Configuration ==================================
    parser.add_argument("--val_split", type=float, default=0.2, help="fraction of labeled data used for validation (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="minimum improvement threshold for early stopping")
    parser.add_argument("--enable_validation", action="store_true", default=False, help="enable validation on labeled data subset")

    # ================================== Data Augmentation and Reproducibility Configuration ==================================
    parser.add_argument("--aug", action="store_true", help="enable data augmentation for unlabeled data")
    parser.add_argument("--no_aug", action="store_true", help="disable data augmentation for unlabeled data")
    parser.add_argument("--freeze", action="store_true", help="freeze randomness for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducible results")

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
    Formula:
        IoU = (pred * mask) / (pred + mask - pred * mask)
        loss = (BCE(pred, mask) + (1 - IoU)) / 2
    """
    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )  # Calculate weighting factor: larger difference between mask and its local average means higher weight, focusing more on edge or transition areas
    weighted_bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    weighted_bce_loss = (weight * weighted_bce_loss).sum(dim=(2, 3)) / weight.sum(
        dim=(2, 3)
    )  # dim=(2, 3) represents summation over spatial dimensions (height and width)

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(
        dim=(2, 3)
    )  # Intersection: product is large only when both are high at corresponding pixels (both have high confidence), reflecting intensity of common activation areas
    union = (((pred + mask - pred * mask)) * weight).sum(dim=(2, 3))  # Represents contribution of prediction and ground truth
    weighted_IoU = (inter + 1e-6) / (union + 1e-6)  # Add 1e-6 to prevent division by zero
    weighted_IoU_loss = 1 - weighted_IoU  # IoU loss: higher IoU means lower loss
    return (weighted_bce_loss + weighted_IoU_loss).mean()


# The following are debugging visualization functions, usually not needed during training
# Uncomment if debugging is needed
"""
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
"""


## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


def load_data(dataset_path, opt, aug=True, freeze=False, dataset_type=""):
    """
    Load dataset (for weaklysupervised learning)
    Args:
        dataset_path: Dataset path
        opt: Training options
        aug: Whether to enable data augmentation
        freeze: Whether to freeze randomness
        dataset_type: Dataset type description
    Returns:
        tuple: (train_loader, total_step)
    """
    image_root = os.path.join(dataset_path, "img/")
    gt_root = os.path.join(dataset_path, "gt/")
    trans_map_root = os.path.join(dataset_path, "trans/")

    train_loader = get_loader(
        image_root,
        gt_root,
        trans_map_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        aug=aug,
        freeze=freeze,
        random_seed=opt.random_seed,
        dataset_type=dataset_type,
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
        aug=False,  # Labeled data does not use data augmentation to maintain stability
        freeze=freeze,
        random_seed=opt.random_seed,
    )

    train_step = len(train_loader)
    val_step = len(val_loader)
    return train_loader, val_loader, train_step, val_step


opt = argparser()

# Handle data augmentation logic - enable augmentation by default unless explicitly disabled
if opt.no_aug:
    opt.aug = False
else:
    opt.aug = True  # Enable data augmentation by default

# Get dataset name and generate model name
labeled_dataset_name = get_dataset_name_from_path(opt.labeled_dataset_path)
unlabeled_dataset_name = get_dataset_name_from_path(opt.unlabeled_dataset_path)
model_name = generate_model_name(labeled_dataset_name, unlabeled_dataset_name, opt.pretrained_weights)
original_save_path = opt.save_model_path


def print_training_configuration(opt, device, labeled_dataset_name, unlabeled_dataset_name, model_name, original_save_path):
    """
    Print training configuration information
    """
    print("=" * 80)
    print("WEAKLY-SUPERVISED TRAINING CONFIGURATION")
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
    print(f"  Generator Learning Rate: {opt.lr_gen}")
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

    # ================================== WEAKLY-Supervised Learning Configuration ==================================
    print("\n WEAKLY-SUPERVISED LEARNING")
    print("-" * 40)
    print(f"  Contrastive Pixel Matching: {'Inter-image' if opt.inter else 'Intra-image'}")
    print(f"  Contrastive Sample Count: {opt.no_samples}")

    # =============== DATASET CONFIGURATION ===============
    print("\nDATASET CONFIGURATION")
    print("-" * 40)
    print(f"  Labeled Dataset: {opt.labeled_dataset_path}")
    print(f"  Labeled Dataset Name: {labeled_dataset_name}")
    print(f"  Unlabeled Dataset: {opt.unlabeled_dataset_path}")
    print(f"  Unlabeled Dataset Name: {unlabeled_dataset_name}")
    print(f"  Pretrained Weights: {opt.pretrained_weights or 'None'}")
    print(f"  Original Save Path: {original_save_path}")
    print(f"  Final Save Path: {opt.save_model_path}")

    # =============== VALIDATION & EARLY STOPPING ===============
    print("\nVALIDATION & EARLY STOPPING")
    print("-" * 40)
    print(f"  Enable Validation: {opt.enable_validation}")
    print(f"  Validation Split: {opt.val_split}")
    print(f"  Early Stopping Patience: {opt.patience}")
    print(f"  Min Delta for Improvement: {opt.min_delta}")

    # =============== DATA AUGMENTATION & REPRODUCIBILITY ===============
    print("\nDATA AUGMENTATION & REPRODUCIBILITY")
    print("-" * 40)
    print(f"  Data Augmentation (unlabeled): {opt.aug}")
    print(f"  Freeze Randomness: {opt.freeze}")
    print(f"  Random Seed: {opt.random_seed}")
    if opt.freeze:
        print("  WARNING: Freeze mode enabled - all randomness frozen for debugging")
        if opt.aug:
            print("  NOTE: Data augmentation disabled due to freeze mode")
    elif opt.aug:
        print("  Data augmentation enabled for unlabeled data")
    else:
        print("  Data augmentation disabled")

    print("=" * 80)


opt.save_model_path = os.path.join(original_save_path, model_name)

# Print the training configuration
print_training_configuration(opt, device, labeled_dataset_name, unlabeled_dataset_name, model_name, original_save_path)

# Build model
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=opt.num_filters)
if opt.pretrained_weights is not None:
    print(f"Load pretrained weights: {opt.pretrained_weights}")
    try:
        checkpoint = torch.load(opt.pretrained_weights, map_location=device)
        generator.load_state_dict(checkpoint)
        print(" Pretrained weights loaded successfully")
    except Exception as e:
        print(f" Failed to load pretrained weights: {e}")
        print("Continuing with random initialization...")

generator.to(device)
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen, betas=(opt.beta, 0.999))

# Load labeled data (with or without validation split)
if opt.enable_validation:
    # Validation mode enabled: use target domain for validation, source domain fully for training
    # Source domain (labeled data): used entirely for training, no split
    train_loader_labeled, total_step_labeled = load_data(
        opt.labeled_dataset_path, opt, aug=False, freeze=opt.freeze, dataset_type="labeled data (full)"
    )

    # Target domain (pseudo-labeled data): split out validation set
    train_loader_un, val_loader, total_step_un, val_step = load_labeled_data_with_validation(opt.unlabeled_dataset_path, opt, freeze=opt.freeze)

    print(f"Labeled training set size (source domain): {total_step_labeled}")
    print(f"Unlabeled training set size (target domain): {total_step_un}")
    print(f"Validation set size (target domain): {val_step}")
    print("Using TARGET DOMAIN for validation!")

    # Initialize early stopping - unified based on training loss
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_train_loss = float("inf")
    best_epoch = 0
    validation_enabled = True
else:
    # Non-validation mode: use all labeled data for training
    train_loader_labeled, total_step_labeled = load_data(opt.labeled_dataset_path, opt, aug=False, freeze=opt.freeze, dataset_type="labeled data")

    # Non-validation mode: use full target domain data
    train_loader_un, total_step_un = load_data(opt.unlabeled_dataset_path, opt, aug=opt.aug, freeze=opt.freeze, dataset_type="unlabeled data")

    val_loader = None
    print(f"Labeled dataset size: {total_step_labeled}")
    print(f"Unlabeled dataset size: {total_step_un}")

    # Initialize early stopping - unified based on training loss
    early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
    best_train_loss = float("inf")
    best_epoch = 0
    validation_enabled = False

train_loader_un_iter = cycle(train_loader_un)  # continuously iterate over the pseudo-labeled dataset

# Use labeled data loader for main training loop
train_loader = train_loader_labeled
total_step = total_step_labeled


# Learning rate scheduler - use ReduceLROnPlateau scheduler, adaptively adjust learning rate based on loss
scheduler = lr_scheduler.ReduceLROnPlateau(
    generator_optimizer,
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
loss_lsc = LocalSaliencyCoherence().to(device)  # Local saliency coherence loss function, enhances prediction consistency in fine-grained regions
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
            gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss + opt.contrastive_loss_weight * cont_loss  # type: torch.Tensor
            gen_loss.backward()

            # Gradient clipping
            if opt.clip > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), opt.clip)

            generator_optimizer.step()

            if rate == 1:
                loss_record.update(gen_loss.item(), opt.batchsize)

            # Release GPU memory
            del gen_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Print training progress at 25%, 50%, 75%, and 100%
        progress_points = [int(total_step * 0.25), int(total_step * 0.5), int(total_step * 0.75), total_step]
        if i in progress_points:
            progress_pct = (i / total_step) * 100
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}] ({:.0f}%), Gen Loss: {:.4f}".format(
                    datetime.now(), epoch, opt.epoch, i, total_step, progress_pct, loss_record.show()
                )
            )

    # Choose scheduler update method based on whether validation is enabled
    old_lr = generator_optimizer.param_groups[0]["lr"]

    if validation_enabled and val_loader is not None:
        # If validation is enabled, call scheduler.step(val_loss) after validation
        pass
    else:
        # If validation is not enabled, use training loss
        current_loss = loss_record.avg
        scheduler.step(current_loss)

    current_lr = generator_optimizer.param_groups[0]["lr"]

    if old_lr != current_lr:
        print(f"Epoch {epoch} completed. Learning rate changed: {old_lr:.6f} -> {current_lr:.6f}")
    else:
        print(f"Epoch {epoch} completed. Learning rate: {current_lr:.6f}")

    # Validation and early stopping logic - unified based on training loss
    current_train_loss = loss_record.avg

    if validation_enabled and val_loader is not None:
        # If validation mode is enabled: evaluate model on validation set (for reference only)
        print("Starting validation...")
        val_loss, val_metrics = validate_model(generator, val_loader, device, structure_loss)

        print(f"Validation Results (Reference Only) - Loss: {val_loss:.4f}")
        print(f"  IoU: {val_metrics['iou']:.4f}")
        print(f"  F1-Score: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

        # Update learning rate scheduler using training loss (not validation loss)
        scheduler.step(current_train_loss)
        new_lr = generator_optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Learning rate adjusted based on training loss: {current_lr:.6f} -> {new_lr:.6f}")

    # Unified model saving and early stopping logic - based on training loss
    if current_train_loss < best_train_loss:
        best_train_loss = current_train_loss
        best_epoch = epoch
        # Save best model
        best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
        torch.save(generator.state_dict(), os.path.join(opt.save_model_path, best_model_filename))
        print(f"New best model saved! Training loss: {current_train_loss:.4f}")
        if validation_enabled and val_loader is not None:
            print(f"  Corresponding validation IoU: {val_metrics['iou']:.4f} (reference)")

    # Unified early stopping check - based on training loss
    early_stopping(current_train_loss, generator)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        print(f"Best training loss: {best_train_loss:.4f} achieved at epoch {best_epoch}")
        break

    # Periodically save checkpoints - use dynamic file names
    if epoch >= 0 and epoch % 10 == 0:
        checkpoint_filename = generate_checkpoint_filename(epoch, model_name, opt.pretrained_weights)
        torch.save(generator.state_dict(), os.path.join(opt.save_model_path, checkpoint_filename))
        print(f"Checkpoint saved: {checkpoint_filename}")

# Summary after training ends
print("\n" + "=" * 50)
print("Weakly-Supervised Training completed!")
print(f"Best training loss: {best_train_loss:.4f} achieved at epoch {best_epoch}")
best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
print(f"Best model saved at: {os.path.join(opt.save_model_path, best_model_filename)}")
print("=" * 50)
