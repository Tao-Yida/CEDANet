import torch
import torch.nn.functional as F
from torch import no_grad

import numpy as np
import os, argparse
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from dataloader import get_train_val_loaders, get_dataset_name_from_path
from utils import AvgMeter, EarlyStopping, validate_model, generate_model_name, generate_checkpoint_filename, generate_best_model_filename
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation
from lscloss import *

# Define computation device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_argparser():
    parser = argparse.ArgumentParser(description="Fully Supervised Training Script")

    # Basic training configuration
    parser.add_argument("--epoch", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=10, help="batch size for training")
    parser.add_argument("--trainsize", type=int, default=352, help="input image resolution (trainsize x trainsize)")

    # Optimizer configuration
    parser.add_argument("--lr_gen", type=float, default=5e-5, help="learning rate for generator")
    parser.add_argument("--beta", type=float, default=0.5, help="beta parameter for Adam optimizer")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="learning rate decay factor for ReduceLROnPlateau")
    parser.add_argument("--decay_epoch", type=int, default=6, help="patience epochs for ReduceLROnPlateau scheduler")

    # Model architecture configuration
    parser.add_argument("--gen_reduced_channel", type=int, default=32, help="reduced channel count in generator")
    parser.add_argument("--feat_channel", type=int, default=32, help="feature channel count for saliency features")
    parser.add_argument("--latent_dim", type=int, default=3, help="latent space dimension")

    # Loss function weight configuration
    parser.add_argument("--reg_weight", type=float, default=1e-4, help="weight for L2 regularization")
    parser.add_argument("--lat_weight", type=float, default=10.0, help="weight for latent loss")
    parser.add_argument("--vae_loss_weight", type=float, default=0.4, help="weight for VAE loss component")

    # Dataset path configuration
    parser.add_argument("--dataset_path", type=str, default="data/ijmond_data/train", help="training dataset path")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="path to pretrained model weights")
    parser.add_argument("--save_model_path", type=str, default="models/full-supervision", help="directory to save trained models")

    # Validation and early stopping configuration
    parser.add_argument("--val_split", type=float, default=0.1, help="fraction of dataset used for validation (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs)")
    parser.add_argument("--min_delta", type=float, default=0.001, help="minimum improvement threshold for early stopping")

    # Data augmentation and reproducibility configuration
    parser.add_argument("--aug", action="store_true", default=False, help="enable data augmentation for training")
    parser.add_argument("--freeze", action="store_true", default=False, help="freeze randomness for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducible results")

    return parser


def print_training_configuration(opt, device, dataset_name, model_name, original_save_path):
    """
    Print training configuration information
    """
    print("=" * 80)
    print("FULLY SUPERVISED TRAINING CONFIGURATION")
    print("=" * 80)

    # Basic configuration
    print("📋 BASIC TRAINING SETTINGS")
    print("-" * 40)
    print(f"  Training Epochs: {opt.epoch}")
    print(f"  Batch Size: {opt.batchsize}")
    print(f"  Training Image Size: {opt.trainsize}x{opt.trainsize}")
    print(f"  Device: {device}")
    print(f"  Dataset Name: {dataset_name}")
    print(f"  Model Name: {model_name}")

    # Optimizer configuration
    print("\n⚙️  OPTIMIZER SETTINGS")
    print("-" * 40)
    print(f"  Generator Learning Rate: {opt.lr_gen}")
    print(f"  Adam Beta: {opt.beta}")
    print(f"  LR Decay Factor: {opt.decay_rate}")
    print(f"  LR Patience (epochs): {opt.decay_epoch}")

    # Model architecture configuration
    print("\n🏗️  MODEL ARCHITECTURE")
    print("-" * 40)
    print(f"  Generator Reduced Channels: {opt.gen_reduced_channel}")
    print(f"  Feature Channels: {opt.feat_channel}")
    print(f"  Latent Dimension: {opt.latent_dim}")

    # Loss function weights
    print("\n📊 LOSS FUNCTION WEIGHTS")
    print("-" * 40)
    print(f"  L2 Regularization: {opt.reg_weight}")
    print(f"  Latent Loss: {opt.lat_weight}")
    print(f"  VAE Loss: {opt.vae_loss_weight}")

    # Dataset configuration
    print("\n📁 DATASET CONFIGURATION")
    print("-" * 40)
    print(f"  Dataset Path: {opt.dataset_path}")
    print(f"  Pretrained Weights: {opt.pretrained_weights or 'None'}")
    print(f"  Original Save Path: {original_save_path}")
    print(f"  Final Save Path: {opt.save_model_path}")

    # Validation and early stopping configuration
    print("\n✅ VALIDATION & EARLY STOPPING")
    print("-" * 40)
    print(f"  Validation Split: {opt.val_split}")
    print(f"  Early Stopping Patience: {opt.patience}")
    print(f"  Min Delta for Improvement: {opt.min_delta}")

    # Data augmentation configuration
    print("\n🔀 DATA AUGMENTATION & REPRODUCIBILITY")
    print("-" * 40)
    print(f"  Data Augmentation: {opt.aug}")
    print(f"  Freeze Randomness: {opt.freeze}")
    print(f"  Random Seed: {opt.random_seed}")
    if opt.freeze and opt.aug:
        print("  NOTE: Data augmentation disabled due to freeze mode")

    print("=" * 80)


parser = create_argparser()

# aug  freeze  effect  usage scenario
# False  False  basic training, no augmentation  quick test
# True   False  normal training, with augmentation  recommended training
# False  True   debug mode, fully fixed  model debugging
# True   True   debug mode, augmentation disabled  debug augmentation logic


# All hyperparameters are stored in opt
opt = parser.parse_args()

# Get dataset name and generate model name
dataset_name = get_dataset_name_from_path(opt.dataset_path)
model_name = generate_model_name(dataset_name, opt.pretrained_weights)
original_save_path = opt.save_model_path
opt.save_model_path = os.path.join(original_save_path, model_name)

# Print training configuration
print_training_configuration(opt, device, dataset_name, model_name, original_save_path)
print("\nData Augmentation & Reproducibility:")
print("  - Data Augmentation: {}".format("Enabled" if opt.aug else "Disabled"))
print("  - Freeze Mode: {}".format("Enabled" if opt.freeze else "Disabled"))
print("  - Random Seed: {}".format(opt.random_seed))
if opt.freeze:
    print("  Freeze mode enabled - all randomness frozen for debugging")
if opt.freeze and opt.aug:
    print("  Data augmentation will be disabled due to freeze mode")
print("==========================================\n")

# Build models
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)  # Generator model

# Load pretrained weights if available, otherwise use random initialization
if opt.pretrained_weights is not None:
    print(f"Load pretrained weights: {opt.pretrained_weights}")
    generator.load_state_dict(torch.load(opt.pretrained_weights))

generator.to(device)  # Move generator model to computation device
generator_params = generator.parameters()  # Get generator model parameters as iterable
generator_optimizer = torch.optim.Adam(
    generator_params, lr=opt.lr_gen, betas=(opt.beta, 0.999)
)  # Adam optimizer, betas control decay rates of first and second moment estimates

gt_root = os.path.join(opt.dataset_path, "gt/")  # data/ijmond_data/test/gt
image_root = os.path.join(opt.dataset_path, "img/")  # data/ijmond_data/test/img
gt_root = os.path.join(opt.dataset_path, "gt/")  # data/ijmond_data/test/gt
trans_map_root = os.path.join(opt.dataset_path, "trans/")  # data/ijmond_data/test/trans

# Get data loaders - using new data augmentation and reproducibility parameters
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
# Calculate total steps of dataset, training set is divided into multiple batches for training
total_step = len(train_loader)
print(f"Training steps per epoch: {total_step}")
print(f"Validation steps per epoch: {len(val_loader)}")

# Initialize early stopping strategy and best model tracking
early_stopping = EarlyStopping(patience=opt.patience, min_delta=opt.min_delta, restore_best_weights=True)
best_val_iou = 0.0
best_epoch = 0

# Learning rate scheduler - use ReduceLROnPlateau scheduler, adaptively adjust learning rate based on loss
scheduler = lr_scheduler.ReduceLROnPlateau(
    generator_optimizer,
    mode="min",  # Monitor loss, reduce learning rate when loss stops decreasing
    factor=opt.decay_rate,  # Learning rate decay factor
    patience=opt.decay_epoch,  # How many epochs to wait if no improvement before reducing learning rate
    min_lr=1e-7,  # Minimum learning rate
)
print(f"Learning Rate Scheduler configured:")
print(f"  - Type: ReduceLROnPlateau (adaptive based on validation loss)")
print(f"  - Patience (epochs to wait): {opt.decay_epoch}")
print(f"  - Decay Factor: {opt.decay_rate}")
print(f"  - Minimum LR: 1e-7")

size_rates = [1]  # Multi-scale training, scale factor, set to 1 means no scaling
lsc_loss = LocalSaliencyCoherence().to(device)  # Local saliency coherence loss function, strengthens prediction consistency in fine-grained regions
lsc_loss_kernels_desc_defaults = [{"weight": 0.1, "xy": 3, "trans": 0.1}]  # Used for kernel function calculation
lsc_loss_radius = 2  # Neighborhood radius
weight_lsc = 0.1  # Controls the weight of local saliency coherence loss in total loss


def structure_loss(pred, mask):
    """
    Structure loss, used to evaluate the difference between the predicted saliency map and the ground truth saliency map
    Achieved by calculating weighted binary cross-entropy loss and weighted IoU loss
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
    )  # Calculate weighting factor, the greater the difference between mask and its local mean, the greater the weight, thus focusing more on edge or transition regions
    weighted_bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    weighted_bce_loss = (weight * weighted_bce_loss).sum(dim=(2, 3)) / weight.sum(
        dim=(2, 3)
    )  # dim=(2, 3) means summing over spatial dimensions, corresponding to height and width

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(
        dim=(2, 3)
    )  # Intersection, only when both are high at corresponding pixels (i.e., both have high confidence), the product is large, reflecting the strength of the common activation region
    union = (((pred + mask - pred * mask)) * weight).sum(dim=(2, 3))  # Represents the contribution of prediction and ground truth respectively
    weighted_IoU = (inter + 1e-6) / (union + 1e-6)  # Add 1e-6 to prevent division by zero
    weighted_IoU_loss = 1 - weighted_IoU  # IoU loss, the higher the IoU, the lower the loss
    return (weighted_bce_loss + weighted_IoU_loss).mean()


def visualize_prediction_init(pred):
    """
    Visualize prediction results
    Args:
        pred: Predicted saliency map, size: [batch_size, channels, height, width]
    """
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk, :, :, :]  # Extract the prediction result of the kk-th image
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0  # Scale prediction result to 0-255 range
        pred_edge_kk = pred_edge_kk.astype(np.uint8)  # Convert to uint8 type
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


def linear_annealing(init, fin, step, annealing_steps):
    """
    Linear annealing of a parameter.
    Args:
        init: initial value
        fin: final value
        step: current step
        annealing_steps: total steps for annealing
    """
    if annealing_steps == 0:  # If no annealing steps are set, return the final value directly
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed


print("Let's go!")
# Ensure the save directory exists before training starts
save_path = opt.save_model_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Created save directory: {save_path}")

for epoch in range(1, (opt.epoch + 1)):
    print("--" * 10 + "Epoch: {}/{}".format(epoch, opt.epoch) + "--" * 10)
    # Remove scheduler.step() here, will be called at the end of the epoch
    generator.train()
    loss_record = AvgMeter()

    # Training phase
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, trans = pack
            # Use device-agnostic .to(device) instead of .cuda()
            images = images.to(device)
            gts = gts.to(device)
            trans = trans.to(device)
            # Multi-scale training samples
            trainsize = int(
                round(opt.trainsize * rate / 32) * 32
            )  # Adjust training size to a multiple of 32, compatible with most networks (up/down sampling operations require input size to be a multiple of 32)
            if rate != 1:  # If not original size, upsample
                images = F.interpolate(images, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode="bilinear", align_corners=True)
                trans = F.interpolate(trans, size=(trainsize, trainsize), mode="bilinear", align_corners=True)

            pred_post_init, pred_post_ref, pred_prior_init, pred_piror_ref, latent_loss = generator.forward(images, gts)

            # Re-scale data for crf loss
            # Downsample to 0.3x original size for CRF loss calculation, improving computation speed
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

            # l2 regularizer the inference model
            reg_loss = l2_regularisation(generator.xy_encoder) + l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
            reg_loss = opt.reg_weight * reg_loss  # Weight the regularization loss, control the weight of regularization loss in total loss
            latent_loss = latent_loss

            sal_loss = 0.5 * (
                structure_loss(pred_post_init, gts) + structure_loss(pred_post_ref, gts)
            )  # Structure loss of two predicted posterior results, measures the difference between prediction and ground truth, considers both pixel-level and region-level accuracy
            anneal_reg = linear_annealing(
                0, 1, epoch, opt.epoch
            )  # Prevent posterior collapse in early training, let the model focus on reconstruction first, then gradually strengthen the constraint on latent distribution
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

            # Conditional variational autoencoder loss, including saliency difference loss, latent space loss, and posterior consistency loss
            gen_loss_cvae = sal_loss + latent_loss + loss_lsc_post
            gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

            # Generator structure loss
            gen_loss_gsnn = 0.5 * (structure_loss(pred_prior_init, gts) + structure_loss(pred_post_ref, gts))
            gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn + loss_lsc_prior
            # total loss
            gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss
            gen_loss.backward()
            generator_optimizer.step()

            if rate == 1:
                loss_record.update(gen_loss.data, opt.batchsize)

        # Print training info - print at 25%, 50%, 75%, 100%
        progress_points = [int(total_step * 0.25), int(total_step * 0.5), int(total_step * 0.75), total_step]
        if i in progress_points:
            progress_pct = (i / total_step) * 100
            # Calculate pixel-level confusion matrix metrics
            with torch.no_grad():
                # Binarize prediction, threshold 0.5
                pred_bin = (torch.sigmoid(pred_post_init) > 0.5).float()
                gt_bin = gts
                # Flatten all pixels
                pred_flat = pred_bin.view(-1)
                gt_flat = gt_bin.view(-1)
                tp = ((pred_flat == 1) & (gt_flat == 1)).sum().item()
                tn = ((pred_flat == 0) & (gt_flat == 0)).sum().item()
                fp = ((pred_flat == 1) & (gt_flat == 0)).sum().item()
                fn = ((pred_flat == 0) & (gt_flat == 1)).sum().item()
            # Print total loss and confusion matrix
            print(
                "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}] ({:.0f}%), Gen Loss: {:.4f}, TP: {}, FP: {}, TN: {}, FN: {}".format(
                    datetime.now(), epoch, opt.epoch, i, total_step, progress_pct, loss_record.show(), tp, fp, tn, fn
                )
            )

    # Validation phase
    print("Starting validation...")
    val_loss, val_metrics = validate_model(generator, val_loader, device, structure_loss)

    print(f"Validation Results - Loss: {val_loss:.4f}")
    print(f"  IoU: {val_metrics['iou']:.4f}")
    print(f"  F1-Score: {val_metrics['f1']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

    # Call scheduler.step() after validation - ReduceLROnPlateau requires the monitored metric
    old_lr = generator_optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)  # Update learning rate using validation loss
    current_lr = generator_optimizer.param_groups[0]["lr"]

    if old_lr != current_lr:
        print(f"Epoch {epoch} completed. Learning rate changed: {old_lr:.6f} -> {current_lr:.6f}")
    else:
        print(f"Epoch {epoch} completed. Learning rate: {current_lr:.6f}")

    # Check if this is the best model - use IoU as the main metric
    current_iou = val_metrics["iou"]
    current_f1 = val_metrics["f1"]
    if current_iou > best_val_iou:
        best_val_iou = current_iou
        best_epoch = epoch
        # Save best model - use dynamic filename
        save_path = opt.save_model_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
        best_model_path = os.path.join(save_path, best_model_filename)
        torch.save(generator.state_dict(), best_model_path)
        print(f"New best model saved! IoU: {current_iou:.4f}, F1: {current_f1:.4f}")
        print(f"   Saved as: {best_model_filename}")

    # Early stopping check - use IoU
    early_stopping(current_iou, generator)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        print(f"Best IoU score: {best_val_iou:.4f} at epoch {best_epoch}")
        break

    # Periodically save checkpoints - use dynamic filename
    save_path = opt.save_model_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch >= 0 and epoch % 5 == 0:
        checkpoint_filename = generate_checkpoint_filename(epoch, model_name, opt.pretrained_weights)
        checkpoint_path = os.path.join(save_path, checkpoint_filename)
        torch.save(generator.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_filename}")

# Summary after training ends
print("\n" + "=" * 50)
print("Training completed!")
print(f"Best validation IoU score: {best_val_iou:.4f} achieved at epoch {best_epoch}")
best_model_filename = generate_best_model_filename(model_name, opt.pretrained_weights)
print(f"Best model saved at: {os.path.join(opt.save_model_path, best_model_filename)}")
print("=" * 50)
