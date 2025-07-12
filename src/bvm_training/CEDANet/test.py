import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os, argparse
from scipy import misc
from model.ResNet_models import Generator
from domain_adapt import create_domain_adaptive_model
from dataloader import test_dataset
import cv2
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_mse(pred, gt):
    """Compute Mean Squared Error (MSE)"""
    pred_norm = pred.astype(np.float32) / 255.0
    gt_norm = gt.astype(np.float32) / 255.0
    return mean_squared_error(gt_norm.flatten(), pred_norm.flatten())


def compute_mae(pred, gt):
    """Compute Mean Absolute Error (MAE)"""
    pred_norm = pred.astype(np.float32) / 255.0
    gt_norm = gt.astype(np.float32) / 255.0
    return mean_absolute_error(gt_norm.flatten(), pred_norm.flatten())


parser = argparse.ArgumentParser()
parser.add_argument("--testsize", type=int, default=352, help="testing size")
parser.add_argument("--latent_dim", type=int, default=8, help="latent dim")
parser.add_argument("--feat_channel", type=int, default=32, help="reduced channel of saliency feat")
parser.add_argument("--model_path", type=str, default="./models/domain_adapt/Model_16_gen.pth", help="path to domain adapted model file")
parser.add_argument("--test_dataset", type=str, default="ijmond", choices=["ijmond", "smoke5k"], help="test dataset: ijmond | smoke5k")
parser.add_argument("--num_domains", type=int, default=2, help="number of domains (source=0, target=1)")
parser.add_argument("--domain_loss_weight", type=float, default=0.5, help="domain loss weight used during training")
parser.add_argument("--use_ldconv", action="store_true", default=False, help="use LDConv in domain discriminators")
parser.add_argument("--use_attention_pool", action="store_true", default=False, help="use AttentionPool2d in domain discriminators")
opt = parser.parse_args()

# Set data path according to the test dataset
if opt.test_dataset == "ijmond":
    dataset_path = "data/ijmond_data/test/img/"
elif opt.test_dataset == "smoke5k":
    dataset_path = "data/SMOKE5K_Dataset/SMOKE5K/test/img/"

# Detect and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Build domain adaptation model
base_generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim, num_filters=16)
model = create_domain_adaptive_model(
    base_generator=base_generator,
    feat_channels=opt.feat_channel,
    num_domains=opt.num_domains,
    domain_loss_weight=opt.domain_loss_weight,
    use_ldconv=opt.use_ldconv,
    use_attention_pool=opt.use_attention_pool,
)

# Robust model loading
print("Loading domain adapted model...")
try:
    if device.type == "cuda":
        checkpoint = torch.load(opt.model_path, weights_only=True)
    else:
        checkpoint = torch.load(opt.model_path, map_location="cpu", weights_only=True)

    # Check if it is a full checkpoint
    if "generator_state_dict" in checkpoint:
        # Load generator weights
        generator_dict = checkpoint["generator_state_dict"]
        base_generator_dict = base_generator.state_dict()
        filtered_dict = {
            k: v for k, v in generator_dict.items() if k in base_generator_dict and generator_dict[k].shape == base_generator_dict[k].shape
        }
        base_generator.load_state_dict(filtered_dict, strict=False)
        print(f"Successfully loaded generator: {len(filtered_dict)}/{len(generator_dict)} parameters")

        # If there are domain adapter weights, try to load them as well
        if "domain_disc_smoke.weight" in checkpoint or "domain_disc_bg.weight" in checkpoint:
            try:
                # Try to load the complete domain adaptation model weights
                filtered_model_dict = {}
                model_state_dict = model.state_dict()
                for k, v in checkpoint.items():
                    if k in model_state_dict and checkpoint[k].shape == model_state_dict[k].shape:
                        filtered_model_dict[k] = v
                model.load_state_dict(filtered_model_dict, strict=False)
                print("Successfully loaded complete domain adaptive model")
            except Exception as e:
                print(f"Warning: Could not load domain adapter weights: {e}, using only generator")
    elif isinstance(checkpoint, dict) and "base_generator.sal_encoder.resnet.conv1.weight" in checkpoint:
        # Checkpoint directly contains the full weights of base_generator
        base_generator_dict = base_generator.state_dict()
        filtered_dict = {}
        for k, v in checkpoint.items():
            # Remove the "base_generator." prefix (if present)
            clean_key = k.replace("base_generator.", "") if k.startswith("base_generator.") else k
            if clean_key in base_generator_dict and checkpoint[k].shape == base_generator_dict[clean_key].shape:
                filtered_dict[clean_key] = v
        base_generator.load_state_dict(filtered_dict, strict=False)
        print(f"Successfully loaded base generator weights: {len(filtered_dict)}/{len(checkpoint)} parameters")
    else:
        # Assume these are pure generator weights
        base_generator_dict = base_generator.state_dict()
        filtered_dict = {k: v for k, v in checkpoint.items() if k in base_generator_dict and checkpoint[k].shape == base_generator_dict[k].shape}
        base_generator.load_state_dict(filtered_dict, strict=False)
        print(f"Successfully loaded generator weights: {len(filtered_dict)}/{len(checkpoint)} parameters")

    if len(filtered_dict if "filtered_dict" in locals() else checkpoint) == 0:
        raise Exception("No matching parameters found")

except Exception as e:
    print(f"Model loading failed: {e}")
    print("Please check if the model file and architecture match")
    exit(1)

model.to(device)
model.eval()

# Add debug info: check model structure
print(f"Domain adaptive model components:")
print(f"  - Base generator parameters: {sum(p.numel() for p in model.base_generator.parameters())}")
print(f"  - Domain discriminator (smoke) parameters: {sum(p.numel() for p in model.domain_disc_smoke.parameters())}")
print(f"  - Domain discriminator (bg) parameters: {sum(p.numel() for p in model.domain_disc_bg.parameters())}")
print(f"  - Total model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"  - Model is on device: {next(model.parameters()).device}")

# Test model inference
print("Testing model inference...")
test_input = torch.randn(1, 3, opt.testsize, opt.testsize).to(device)
try:
    with torch.no_grad():
        test_output = model(test_input, training=False)
    print(f"Model inference test successful. Output shape: {test_output.shape}")
except Exception as e:
    print(f"Model inference test failed: {e}")
    print("Please check the model architecture compatibility")
    exit(1)

# Initialize cumulative metrics
sum_TP = 0
sum_FP = 0
sum_FN = 0
total_images = 0

# test_datasets = ['CAMO','CHAMELEON','COD10K']
test_datasets = [""]


def compute_energy(disc_score):
    if opt.energy_form == "tanh":
        energy = torch.tanh(-disc_score.squeeze())
    elif opt.energy_form == "sigmoid":
        energy = F.sigmoid(-disc_score.squeeze())
    elif opt.energy_form == "identity":
        energy = -disc_score.squeeze()
    elif opt.energy_form == "softplus":
        energy = F.softplus(-disc_score.squeeze())
    return energy


for dataset in test_datasets:
    # Extract model info from model path
    model_name = os.path.splitext(os.path.basename(opt.model_path))[0]  # File name without extension
    model_dir = os.path.basename(os.path.dirname(opt.model_path))  # Parent directory name

    # Build save path, including model info and domain adaptation identifier
    save_path = os.path.join("./results", "thesis", opt.test_dataset, model_dir, model_name, dataset)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = os.path.join(dataset_path, dataset) if dataset else dataset_path
    print(f"Loading test data from: {image_root}")

    # Check if path exists
    if not os.path.exists(image_root):
        print(f"Warning: Path {image_root} does not exist, skipping...")
        continue

    # Determine GT path
    gt_root = None
    if opt.test_dataset == "ijmond":
        gt_root = os.path.join("data/ijmond_data/test/gt", dataset) if dataset else "data/ijmond_data/test/gt/"
    elif opt.test_dataset == "smoke5k":
        gt_root = os.path.join("data/SMOKE5K_Dataset/SMOKE5K/test/gt_", dataset) if dataset else "data/SMOKE5K_Dataset/SMOKE5K/test/gt_/"

    # Check if GT path exists
    if gt_root and not os.path.exists(gt_root):
        print(f"Warning: GT path {gt_root} does not exist, evaluation metrics will not be calculated")
        gt_root = None

    # Initialize evaluation metric statistics variables
    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    sum_TN = 0
    total_images = 0

    # Initialize additional evaluation metric accumulation variables
    sum_mse = 0.0
    sum_mae = 0.0

    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(f"Processing image {i+1}/{test_loader.size}")
        image, HH, WW, name = test_loader.load_data()
        image = image.to(device)

        # Use domain adaptation model for prediction
        with torch.no_grad():
            # Forward pass of domain adaptation model - only returns final prediction in inference mode
            generator_pred = model(image, training=False)
            # In inference mode, the domain adaptation model calls the base generator and only returns self.prob_pred

        res = generator_pred
        res = F.interpolate(res, size=[WW, HH], mode="bilinear", align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)

        # Safe file saving
        save_file_path = os.path.join(save_path, name)
        try:
            cv2.imwrite(save_file_path, res)
        except Exception as e:
            print(f"Failed to save {save_file_path}: {e}")

        # Compute evaluation metrics (if GT path exists)
        if gt_root is not None:
            gt_path = os.path.join(gt_root, name)
            if os.path.exists(gt_path):
                # Read GT image
                gt_mask = cv2.imread(gt_path, 0)
                if gt_mask is not None:
                    # Binarize GT and prediction
                    gt_bin = (gt_mask > 128).astype(np.uint8)
                    pred_bin = (res > 128).astype(np.uint8)

                    # Compute TP, FP, FN, TN
                    TP = np.logical_and(pred_bin, gt_bin).sum()
                    FP = np.logical_and(pred_bin, 1 - gt_bin).sum()
                    FN = np.logical_and(1 - pred_bin, gt_bin).sum()
                    TN = np.logical_and(1 - pred_bin, 1 - gt_bin).sum()

                    sum_TP += TP
                    sum_FP += FP
                    sum_FN += FN
                    sum_TN += TN

                    # Compute additional evaluation metrics
                    mse = compute_mse(res, gt_mask)
                    mae = compute_mae(res, gt_mask)

                    sum_mse += mse
                    sum_mae += mae

                    total_images += 1
            else:
                print(f"Warning: GT file {gt_path} not found")

    # Compute and save evaluation metrics
    if total_images > 0:
        # Compute metrics
        precision = sum_TP / (sum_TP + sum_FP + 1e-8)
        recall = sum_TP / (sum_TP + sum_FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = sum_TN / (sum_TN + sum_FP + 1e-8)
        accuracy = (sum_TP + sum_TN) / (sum_TP + sum_TN + sum_FP + sum_FN + 1e-8)

        # IoU calculation
        iou_positive = sum_TP / (sum_TP + sum_FP + sum_FN + 1e-8)  # Smoke class IoU
        iou_negative = sum_TN / (sum_TN + sum_FN + sum_FP + 1e-8)  # Background class IoU
        miou = (iou_positive + iou_negative) / 2  # True mIoU: average of two class IoUs

        # Dice coefficient (equivalent to F1-Score)
        dice = 2 * sum_TP / (2 * sum_TP + sum_FP + sum_FN + 1e-8)

        # Compute mean additional metrics
        mean_mse = sum_mse / total_images
        mean_mae = sum_mae / total_images

        # Print results
        print(f"\n=== Domain Adaptation Evaluation Results for {opt.test_dataset} dataset ===")
        print(f"Processed {total_images} images")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"IoU (Smoke): {iou_positive:.4f}")
        print(f"IoU (Background): {iou_negative:.4f}")
        print(f"mIoU: {miou:.4f}")
        print(f"Dice Coefficient: {dice:.4f}")
        print(f"\n=== Additional Evaluation Metrics ===")
        print(f"Mean MSE: {mean_mse:.6f}")
        print(f"Mean MAE: {mean_mae:.6f}")

        # Confusion matrix
        confusion_matrix = f"""
Confusion Matrix:
                Predicted
                 P    N
Actual P    {sum_TP:8d} {sum_FN:8d}
Actual N    {sum_FP:8d} {sum_TN:8d}
"""
        print(confusion_matrix)

        # Save evaluation metrics to txt file
        metrics_file = os.path.join(save_path, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Domain Adaptation Evaluation Results for {opt.test_dataset} dataset\n")
            f.write(f"Test Dataset: {opt.test_dataset}\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write(f"GT Path: {gt_root}\n")
            f.write(f"Model: {opt.model_path}\n")
            f.write(f"Method: domain_adaptation_thesis\n")
            f.write(f"Test size: {opt.testsize}\n")
            f.write(f"Number of domains: {opt.num_domains}\n")
            f.write(f"Domain loss weight: {opt.domain_loss_weight}\n")
            f.write(f"Total images processed: {total_images}\n\n")

            f.write("Evaluation Metrics:\n")
            f.write(f"Precision: {precision:.6f}\n")
            f.write(f"Recall: {recall:.6f}\n")
            f.write(f"F1-Score: {f1_score:.6f}\n")
            f.write(f"Specificity: {specificity:.6f}\n")
            f.write(f"Accuracy: {accuracy:.6f}\n")
            f.write(f"IoU (Smoke): {iou_positive:.6f}\n")
            f.write(f"IoU (Background): {iou_negative:.6f}\n")
            f.write(f"mIoU: {miou:.6f}\n")
            f.write(f"Dice Coefficient: {dice:.6f}\n\n")

            f.write("Additional Evaluation Metrics:\n")
            f.write(f"Mean MSE: {mean_mse:.8f}\n")
            f.write(f"Mean MAE: {mean_mae:.8f}\n\n")

            f.write("Confusion Matrix:\n")
            f.write("                Predicted\n")
            f.write("                 P    N\n")
            f.write(f"Actual P    {sum_TP:8d} {sum_FN:8d}\n")
            f.write(f"Actual N    {sum_FP:8d} {sum_TN:8d}\n\n")

            f.write("Raw counts:\n")
            f.write(f"True Positives (TP): {sum_TP}\n")
            f.write(f"False Positives (FP): {sum_FP}\n")
            f.write(f"False Negatives (FN): {sum_FN}\n")
            f.write(f"True Negatives (TN): {sum_TN}\n")

        print(f"Evaluation metrics saved to: {metrics_file}")
    else:
        print("No valid GT images found for evaluation")
